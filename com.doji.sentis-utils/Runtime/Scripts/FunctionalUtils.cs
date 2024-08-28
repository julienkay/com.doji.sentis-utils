using Doji.AI;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Xml.Linq;
using Unity.Sentis;
using UnityEngine.Assertions;

public static class FunctionalUtils {

    private static FunctionalTensor[] _tmpTensorRefs = new FunctionalTensor[2];

    public static TensorShape shape(this FunctionalTensor tensor) {
        Type tensorType = tensor.GetType();
        PropertyInfo shapeProperty = tensorType.GetProperty("shape", BindingFlags.NonPublic | BindingFlags.Instance);
        if (shapeProperty == null) {
            throw new InvalidOperationException("The 'shape' property was not found.");
        }
        TensorShape shapeValue = (TensorShape)shapeProperty.GetValue(tensor);
        return shapeValue;
    }

    /// <summary>
    /// Alias for <see cref="Ops.Concat(Tensor[], int)"/> 
    /// for convenience of not needing to create a Tensor array.
    /// </summary>
    public static FunctionalTensor Concat(FunctionalTensor tensor1, FunctionalTensor tensor2, int dim = 0) {
        _tmpTensorRefs[0] = tensor1;
        _tmpTensorRefs[1] = tensor2;
        return Functional.Concat(_tmpTensorRefs, dim);
    }

    /// <summary>
    /// Alias for <see cref="Ops.Concat(Tensor[], int)"/> to match torch.cat()
    /// naming and for convenience of not needing to create a Tensor array.
    /// </summary>
    public static FunctionalTensor Cat(FunctionalTensor tensor1, FunctionalTensor tensor2, int dim = 0) {
        _tmpTensorRefs[0] = tensor1;
        _tmpTensorRefs[1] = tensor2;
        return Functional.Concat(_tmpTensorRefs, dim);
    }

    /// <summary>
    /// A List<TensorInt> overload for Cat().
    /// </summary>
    public static FunctionalTensor Cat(List<FunctionalTensor> tensors, int dim = 0) {
        return Functional.Concat(tensors.ToArray(), dim);
    }

    /// <summary>
    /// Fills elements of <paramref name="tensor"/> with <paramref name="value"/> where <paramref name="mask"/> is 1
    /// </summary>
    public static FunctionalTensor MaskedFill(FunctionalTensor tensor, FunctionalTensor mask, float value) {
        FunctionalTensor A = Functional.Constant(value);
        return Functional.Where(mask, A, tensor);
    }

    /// <summary>
    /// Creates a tensor of the given <paramref name="shape"/> size filled with <paramref name="fillValue"/>.
    /// </summary>
    public static FunctionalTensor Full(TensorShape shape, int fillValue) {
        return Functional.Full(shape.ToArray(), fillValue);
    }

    /// <summary>
    /// Creates a 1D tensor of the given <paramref name="size"/> size filled with <paramref name="fillValue"/>.
    /// </summary>
    public static FunctionalTensor Full(int size, int fillValue) {
        return Functional.Full(new int[] { size }, fillValue);
    }

    /// <summary>
    /// Creates a tensor of the given <paramref name="shape"/> size filled with <paramref name="fillValue"/>.
    /// </summary>
    public static FunctionalTensor Full(TensorShape shape, bool fillValue) {
        return Functional.Full(shape.ToArray(), fillValue ? 1 : 0);
    }

    /// <summary>
    /// Creates a 1D tensor of the given <paramref name="size"/> size filled with <paramref name="fillValue"/>.
    /// </summary>
    public static FunctionalTensor Full(int size, bool fillValue) {
        return Functional.Full(new int[] { size }, fillValue ? 1 : 0);
    }

    /// <summary>
    /// Creates a tensor of the given <paramref name="shape"/> size filled with <paramref name="fillValue"/>.
    /// </summary>
    public static FunctionalTensor Full(TensorShape shape, float fillValue) {
        return Functional.Full(shape.ToArray(), fillValue);
    }

    /// <summary>
    /// Creates a 1D tensor of the given <paramref name="size"/> size filled with <paramref name="fillValue"/>.
    /// </summary>
    public static FunctionalTensor Full(int size, float fillValue) {
        return Functional.Full(new int[] { size }, fillValue);
    }

    public static FunctionalTensor Zeros<T>(TensorShape tensorShape) where T : unmanaged {
        switch (typeof(T)) {
            case Type floatType when floatType == typeof(float):
                return Functional.Full(tensorShape.ToArray(), 0f);
            case Type intType when intType == typeof(int):
                return Functional.Full(tensorShape.ToArray(), 0);
            case Type shortType when shortType == typeof(short):
            case Type byteType when byteType == typeof(byte):
                throw new NotImplementedException();
            default:
                throw new ArgumentException($"Invalid tensor data type '{typeof(T)}'");
        }
    }

    public static FunctionalTensor Expand(this FunctionalTensor tensor, TensorShape tensorShape) {
        //TODO: Can we use slicing notation here?
        throw new NotImplementedException();
    }
}
