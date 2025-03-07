using AnomalyDetection.Client.Pages;
using AnomalyDetection.Components;
using AnomalyDetection.Core;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.Extensions.DependencyInjection.Extensions;
using MudBlazor.Services;

namespace AnomalyDetection
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            builder.Services
                .AddRazorComponents()
                .AddInteractiveServerComponents()
                .AddInteractiveWebAssemblyComponents();

            builder.Services.AddMudServices();

            builder.Services
                .AddScoped<DataConfiguration>(svc => new(svc.GetRequiredService<IWebHostEnvironment>().WebRootPath))
                .RegisterAnomalyDetection();
            
            var app = builder.Build();
            
            if (app.Environment.IsDevelopment())
            {
                app.UseWebAssemblyDebugging();
            }
            else
            {
                app.UseExceptionHandler("/Error");
                app.UseHsts();
            }

            app.UseHttpsRedirection();

            app.UseStaticFiles();
            
            app.UseAntiforgery();
            app.MapStaticAssets();
            app
                .MapRazorComponents<App>()
                .AddInteractiveServerRenderMode()
                .AddInteractiveWebAssemblyRenderMode()
                .AddAdditionalAssemblies(typeof(DataVisualize).Assembly);

            app.Run();
        }
        

    }
}
