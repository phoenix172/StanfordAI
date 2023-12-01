using AnomalyDetection.Client.Business;
using AnomalyDetection.Client.ServiceContracts;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using MudBlazor.Services;

var builder = WebAssemblyHostBuilder.CreateDefault(args);

builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });
builder.Services.AddScoped<IAnomalyDetector, AnomalyDetector>();
builder.Services.AddScoped<IMatrixLoader, CsvMatrixLoader>();

builder.Services.AddMudServices();

await builder.Build().RunAsync();
