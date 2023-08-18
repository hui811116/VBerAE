# VBerAE
Variational Bernoulli AutoEncoder

Implement the Bernoulli representation encoders

With training phase using Sigmoid function for smooth gradient.

The main feature is an inverse cumulative density function (CDF) sampling trick.

Where the uniformly [0,1] drawn samples are compared with Bernoulli logits

The obtained representations are random bitstreams. Avoiding the need for quantization.

