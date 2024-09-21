using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Unity.Sentis;

namespace Doji.AI {

    /// <summary>
    /// A class that wraps Sentis' internal IBackend classes with a proxy.
    /// </summary>
    public class BackendProxy : DispatchProxy {
        
        internal object _backendInstance;
        private Type _type;
        private BackendType _backendType;

        public static IBackend Create(BackendType backendType) {
            object backendInstance = CreateBackend(backendType);
            var proxy = Create<IBackend, BackendProxy>();
            (proxy as BackendProxy)._backendInstance = backendInstance;
            (proxy as BackendProxy)._type = backendInstance.GetType();
            (proxy as BackendProxy)._backendType = backendType;
            return proxy;
        }

        protected override object Invoke(MethodInfo targetMethod, object[] args) {
            // Prepare to match parameter types
            Type[] parameterTypes = targetMethod.GetParameters()
                                                .Select(param => param.ParameterType)
                                                .ToArray();

            // Transform args to match the expected parameter types
            for (int i = 0; i < args.Length; i++) {
                if (parameterTypes[i].IsByRefLike && parameterTypes[i].IsGenericType &&
                    parameterTypes[i].GetGenericTypeDefinition() == typeof(ReadOnlySpan<>)) {
                    // Convert the argument to ReadOnlySpan<T>
                    var spanType = parameterTypes[i];
                    var elementType = spanType.GetGenericArguments()[0];
                    Array array = Array.CreateInstance(elementType, ((Array)args[i]).Length);

                    for (int j = 0; j < ((Array)args[i]).Length; j++) {
                        array.SetValue(((Array)args[i]).GetValue(j), j);
                    }

                    args[i] = Activator.CreateInstance(spanType, array);
                }
            }

            // Map methods from our IBackend to Sentis' internal IBackend methods
            var correspondingMethod = _type.GetMethod(targetMethod.Name, parameterTypes);

            if (correspondingMethod != null) {
                return correspondingMethod.Invoke(_backendInstance, args);
            }

            throw new NotImplementedException($"Method {targetMethod.Name} with the specified parameters is not implemented.");
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

        public void ExecuteCommandBufferAndClear() {
            if (_backendType != BackendType.GPUCompute) {
                throw new Exception($"Not possible for {_backendType}");
            }

            Assembly assembly = Assembly.Load("Unity.Sentis");
            Type backendType = assembly.GetType("Unity.Sentis.GPUComputeBackend");

            MethodInfo concatMethod = backendType.GetMethod("ExecuteCommandBufferAndClear", BindingFlags.Public | BindingFlags.Instance);
            concatMethod.Invoke(_backendInstance, null);
        }
    }
}