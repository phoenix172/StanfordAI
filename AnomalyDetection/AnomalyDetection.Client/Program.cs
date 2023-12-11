using AnomalyDetection.Client.Business;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using MudBlazor.Services;

namespace AnomalyDetection.Client;

internal class Program
{
    public static async Task Main(string[] args)
    {
        var builder = WebAssemblyHostBuilder.CreateDefault(args);
        builder.Services.RegisterServices();

        builder.Services.AddMudServices();

        await builder.Build().RunAsync();
    }
}