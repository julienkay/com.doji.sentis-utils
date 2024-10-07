using System.Collections.Generic;
using Unity.Sentis;

namespace Doji.AI {
    public static class WorkerUtils {
        public static void Schedule(this Worker worker, Dictionary<string, Tensor> inputs) {
            foreach (var input in inputs) {
                worker.SetInput(input.Key, input.Value);
            }
            worker.Schedule();
        }
        public static System.Collections.IEnumerator ScheduleIterable(this Worker worker, Dictionary<string, Tensor> inputs) {
            foreach (var input in inputs) {
                worker.SetInput(input.Key, input.Value);
            }
            return worker.ScheduleIterable();
        }
    }
}