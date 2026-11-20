## Intro Summarized
- inference from autoregressive models like transformers in slow -decoding K tokens take K serial runs of the model
- Speculative decoding is an algorithm to sample from autoregressive models faster without any changes to the outputs, by computing tokens in parallel
- observation : inference from larger models is often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication.
- **Speculative execution** is an optimization technique where a task is performed in parallel to verifying if its actually needed
- the above leads to increased concurrecny
- **main contributions**
  - speculative sampling to accelerate sampling from autoregressive models
  - decoding mechanism - speculative decoding that accelerates decoding from autoregressive models

## **Speculative Decoding**
If M<sub>p</sub> is the target model and p( x<sub> t </sub> | x<sub> < t </sub>) is the distribution we get from the model for a prefix  x<sub> < t </sub>. Let M<sub>q</sub> be a more efficient approximation model for the same task and denoted by q(x<sub>t</sub> | x<sub> < t </sub> ) the distribution we get from the model for a prefix. 

<em> 
x<sub> < t </sub> - all tokens before current token t



q(x<sub>t</sub> | x<sub> < t </sub> ) - probability distribution over all possible next tokens, given the previous tokens.
</em>  
The Idea is to use more efficient model MQ to generate completions y belong to Z<sup>+</sup> then use the target model M<sub> p </sub> to evaluate all of the guesses and their respective probabilities from M<sub>q</sub> in parallel accepting those can have ideal solutions and sampling an additional token from an adjusted disribution to fix the first one that was rejected or to add an additional one if they are all accepted

### Standardized Sampling
all popular sampling methods rop-k argmax can easily be cast into a standard sampling from an adjusted distribution eg. argmax sampling is equivalent to zeroing out non max elements of the distribution and normalizing


### Speculative Sampling
to sample x ~ p(x) we instead sample x ~ q(x) keeping q(x) <= p(x) and q(x) > p(x) we reject the sample with probability 1 - p(x) / q(x) and sample x again from an adjusted distribution  p'(x) = norm(max(0, p(x) - q(x))) instead.
