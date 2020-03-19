using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using IoTHubTrigger = Microsoft.Azure.WebJobs.EventHubTriggerAttribute;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.EventHubs;
using System.Text;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.WindowsAzure.Storage.Blob;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace MLOps.StreamAnalytics
{
    public static class IncomingHandler
    {
        private static readonly Lazy<HttpClient> HttpClient = new Lazy<HttpClient>(() =>
        {
            var apiKey = Environment.GetEnvironmentVariable("ApiKey");
            var handler = new HttpClientHandler
            {
                ClientCertificateOptions = ClientCertificateOption.Manual,
                ServerCertificateCustomValidationCallback = (httpRequestMessage, cert, cetChain, policyErrors) => true
            };
            var client = new HttpClient(handler);
            client.DefaultRequestHeaders.TryAddWithoutValidation("Authorization", $"Bearer {apiKey}");
            return client;
        });

        private static readonly string[] IgnoredProperties = {"date", "% Iron Feed", "% Silica Feed", "% Iron Concentrate"};

        [FunctionName(nameof(IncomingHandler))]
        [return: EventHub("mlops-powerbi", Connection = "EventHub")]
        public static async Task<string> Run(
            [IoTHubTrigger("events/messages", Connection = "IoTHub")]  EventData message,
            [Blob("incoming", FileAccess.Write, Connection = "MLOps")] CloudBlobContainer outputContainer,
            ILogger log)
        {
            var uuid = $"p-{message.SystemProperties.PartitionKey}-o-{message.SystemProperties.Offset}";
            
            var apiUrl = Environment.GetEnvironmentVariable("ApiUrl");
            
            string body = Encoding.UTF8.GetString(message.Body.Array);
            JObject jsonContent = JsonConvert.DeserializeObject<JObject>(body);

            List<decimal> properties = jsonContent
                .Properties()
                .Where(property =>  IgnoredProperties.All(ignoredProperty => !ignoredProperty.Equals(property.Name, StringComparison.OrdinalIgnoreCase)) )
                .Select(property => decimal.Parse(property.Value.ToString().Replace(",", ".")))
                .ToList();

            var scoringValue = await GetScore(properties, apiUrl);
            jsonContent.Add("% Silica Concentrate", scoringValue);
            var jsonContentStringified = await UploadToColdPath(outputContainer, uuid, jsonContent);
            log.Log(LogLevel.Debug, $"Uploaded file {uuid} with content {jsonContentStringified}");
            message.Dispose();
            return jsonContentStringified;
        }

        private static async Task<string> UploadToColdPath(CloudBlobContainer outputContainer, string uuid, JObject jsonContent)
        {
            string path = DateTime.UtcNow.ToString("yyyy-MM-dd");
            var directory = outputContainer.GetDirectoryReference(path);
            var blob = directory.GetBlockBlobReference($"{uuid}.json");
            string jsonContentStringified = JsonConvert.SerializeObject(jsonContent);
            await blob.UploadTextAsync(jsonContentStringified);
            return jsonContentStringified;
        }

        private static async Task<decimal> GetScore(IEnumerable<decimal> properties, string apiUrl)
        {
            var stringifiedProperties = String.Join(",", properties);
            var bodyContentToScore = $"{{\"data\": [[{stringifiedProperties}]]}}";
            var content = new StringContent(bodyContentToScore, Encoding.UTF8, "application/json");
            var response = await HttpClient.Value.PostAsync(apiUrl, content);
            var responseContent = await response.Content.ReadAsStringAsync();

            decimal scoringValue = decimal.Parse(responseContent.TrimStart('[').TrimEnd(']'));
            return scoringValue;
        }
    }
}