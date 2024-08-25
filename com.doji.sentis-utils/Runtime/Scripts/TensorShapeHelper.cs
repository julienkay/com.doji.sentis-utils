using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.Sentis;
using UnityEngine;

namespace Doji.AI {

    internal static class TensorShapeHelper {

        public static TensorShape ConcatShape<T>(T[] tensors, int axis) where T : Tensor {
            TensorShape output = tensors[0].shape;

            for (int i = 1; i < tensors.Length; ++i) {
                TensorShape shape = tensors[i].shape;
                output = output.Concat(shape, axis);
            }

            return output;
        }

        public static TensorShape ConcatShape<T>(List<T> tensors, int axis) where T : Tensor {
            TensorShape output = tensors[0].shape;

            for (int i = 1; i < tensors.Count; ++i) {
                TensorShape shape = tensors[i].shape;
                output = output.Concat(shape, axis);
            }

            return output;
        }

        public static TensorShape ConcatShape(Tensor tensor1, Tensor tensor2, int axis) {
            return tensor1.shape.Concat(tensor2.shape, axis);
        }

        public static TensorShape BroadcastShape(Tensor a, Tensor b) {
            return a.shape.Broadcast(b.shape);
        }

        /// <summary>
        /// Creates a `TensorShape` that results from slicing `this` along given axes with given starts, ends, and steps.
        /// </summary>
        /// <param name="starts">The start indices along each of the `axes`.</param>
        /// <param name="ends">The end indices along each of the `axes`.</param>
        /// <param name="axes">The axes along which to slice.</param>
        /// <param name="steps">The step sizes for each of the `axes`.</param>
        /// <returns>The sliced tensor shape.</returns>
        public static TensorShape Slice(this TensorShape shape, ReadOnlySpan<int> starts, ReadOnlySpan<int> ends, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps) {
            Debug.Assert(starts.Length == ends.Length, string.Format("ValueError: starts and ends length do not match {0}, {1}", starts.Length, ends.Length));
             Debug.Assert(starts.Length == axes.Length, string.Format("ValueError: starts and axes length do not match {0}, {1}", starts.Length, axes.Length));
            Debug.Assert(starts.Length == steps.Length, string.Format("ValueError: starts and steps length do not match {0}, {1}", starts.Length, steps.Length));

            TensorShape strided = shape;
            unsafe {
                throw new NotImplementedException();
                /*int* dst = &strided.m_D7;

                for (int i = 0; i < starts.Length; i++) {
                    int outputDim = (int)Math.Ceiling((double)(ends[i] - starts[i]) / (double)steps[i]);
                    dst[(TensorShape.maxRank - shape.rank) + axes[i]] = Mathf.Max(outputDim, 0);
                }*/
            }

            Type tensorShapeType = typeof(TensorShape);
            MethodInfo recomputeMethod = tensorShapeType.GetMethod("RecomputeLength", BindingFlags.NonPublic | BindingFlags.Instance);
            object[] parameters = { };
            recomputeMethod.Invoke(strided, parameters);
            return strided;
        }

        public static TensorShape Concat(this TensorShape shape, TensorShape other, int axis) {
            Type tensorShapeType = typeof(TensorShape);
            MethodInfo concatMethod = tensorShapeType.GetMethod("Concat", BindingFlags.NonPublic | BindingFlags.Instance);
            object[] parameters = { other, axis };
            TensorShape result = (TensorShape)concatMethod.Invoke(shape, parameters);
            return result;
        }

        public static TensorShape Tile(this TensorShape shape, ReadOnlySpan<int> repeats) {
            throw new NotImplementedException();
        }

        public static TensorShape Transpose(this TensorShape shape, ReadOnlySpan<int> premutations) {
            throw new NotImplementedException();
        }
    }
}