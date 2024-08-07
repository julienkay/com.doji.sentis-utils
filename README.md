<a href="https://www.doji-tech.com/">
  <img src="https://www.doji-tech.com/assets/favicon.ico" alt="doji logo" title="Doji" align="right" height="70" />
</a>

# Sentis Utils
A small collection of utility scripts and extensions for Unity Sentis

[OpenUPM]

## About

- Brings back an Ops class similar to what was available in previous Sentis versions. Unlike with the IBackend class it has the convenience of not needing to figure out the shape and allocate the output tensor when doing operations. In terms of memory management it will not reuse any memory and the consumer is expected to call Flush() to clear the memory of all the temporary tensors at a time they see fit.

- It also includes some more operations that are either not implemented in Sentis or could do with an API that is more convenient or more pytorch-like for easier porting of pytorch-based pre-/postprocessing code. This includes methods like Concat, RepeatInterleaved, python-like slicing operators, etc.

**Used by**
- [com.doji.transformers]
- [com.doji.diffusers]

[OpenUPM]: https://openupm.com/packages/com.doji.sentis-utils
[com.doji.transformers]: https://github.com/julienkay/com.doji.diffusers
[com.doji.diffusers]: https://github.com/julienkay/com.doji.diffusers
