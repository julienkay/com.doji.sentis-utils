using System;
using System.Reflection;
using Unity.Sentis;

namespace Doji.AI {

    internal static class ShapeInference {

        public static TensorShape Gather(TensorShape shape, TensorShape indices, int axis) {
            TensorShape gathered = new TensorShape();
            gathered = gathered.BroadcastToRank(shape.rank - 1 + indices.rank);

            if (gathered.rank == 0)
                return gathered;

            axis = shape.Axis(axis);

            for (int i = 0; i < axis; i++)
                gathered[i] = shape[i];
            for (int i = 0; i < indices.rank; i++)
                gathered[axis + i] = indices[i];
            for (int i = axis + 1; i < shape.rank; i++)
                gathered[i + indices.rank - 1] = shape[i];

            return gathered;
        }
    }
}
