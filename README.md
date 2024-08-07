<a href="https://www.doji-tech.com/">
  <img src="https://www.doji-tech.com/assets/favicon.ico" alt="doji logo" title="Doji" align="right" height="70" />
</a>

# Sentis Utils
A small collection of utility scripts and extensions for Unity Sentis

[OpenUPM]

## About

- Brings back an Ops class similar to what was available in previous Sentis versions. Unlike with the IBackend class it has the convenience of not needing to figure out the shape and allocate the output tensor when doing operations. In terms of memory management it will not reuse any memory and the consumer is responsible to call Flush() to free the memory of all the temporary tensors.

- It also includes some more operations that are either not implemented in Sentis or could do with an API that is more convenient or more pytorch-like for easier porting of pytorch-based pre-/postprocessing code. 
  - Quantile()
  - Sort()
  - Concatenate() as an alias for Concat()
  - overloads for Concatenate that take individual Tensors instead of arrays (e.g. Concat(t1, t1))
  - RepeatInterleaved (although not implemented for all shapes yet)
  - Split() that matches the arguments of numpy.split or torch.chunk (i.e. providing a sections parameter rather than requiring start & end parameter). It also takes a preallocated list to store the resulting chunks in.
  - SplitHalf(): returns the two output tensors as a tuple.
  - python-like slicing operators: Overloads for the Slice() method that use C# indices and ranges rather than ReadOnlySpan. When you would do tensor[:, -1, :] with a pytorch tensor you can now do ops.Slice(tensor, .., ^1, ..)

Some not-so-great aspects:
- need to use Reflection to access the internal TensorShape.Split() method to get the shape when splitting tensors

**Used by**
- [com.doji.transformers]
- [com.doji.diffusers]

[OpenUPM]: https://openupm.com/packages/com.doji.sentis-utils
[com.doji.transformers]: https://github.com/julienkay/com.doji.diffusers
[com.doji.diffusers]: https://github.com/julienkay/com.doji.diffusers
