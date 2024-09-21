using System;
using System.Linq;
using System.Reflection;
using Unity.Sentis;

namespace Doji.AI {

    /// <summary>
    /// A class that wraps Sentis' internal IBackend classes with a proxy.
    /// </summary>
    public class BackendProxy : DispatchProxy {
        private object _backendInstance;
        private Type _backendType;

        public static IBackend Create(BackendType backendType) {
            object backendInstance = CreateBackend(backendType);
            var proxy = Create<IBackend, BackendProxy>();
            (proxy as BackendProxy)._backendInstance = backendInstance;
            (proxy as BackendProxy)._backendType = backendInstance.GetType();
            return proxy;
        }

        protected override object Invoke(MethodInfo targetMethod, object[] args) {
            Type[] parameterTypes = args.Select(arg => arg.GetType()).ToArray();

            // Map methods from our IBackend to Sentis' internal IBackend methods
            var correspondingMethod = _backendType.GetMethod(targetMethod.Name, parameterTypes);
            if (correspondingMethod != null) {
                return correspondingMethod.Invoke(_backendInstance, args);
            }
            throw new NotImplementedException($"Method {targetMethod.Name} is not implemented.");
        }

        private static object CreateBackend(BackendType backend) {
            Assembly assembly = Assembly.Load("Unity.Sentis");

            Type backendType = null;
            backendType = backend switch {
                BackendType.GPUCompute => assembly.GetType("Unity.Sentis.GPUComputeBackend"),
                BackendType.GPUPixel => assembly.GetType("Unity.Sentis.GPUPixelBackend"),
                _ => assembly.GetType("Unity.Sentis.CPUBackend"),
            };
            return Activator.CreateInstance(backendType);
        }

    }
}