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

- given distribution q(x) obtained from running Mq on a conditioning prefix, we can sample a token x<sub>1</sub> ~ q(x).

- the prefix may look like - `The cat is` tokens can further sampled  by the draft model - `sitting` then further it comes q(`The cat is sitting`).

- Given the distribution q(x) obtained from running M<sub>q</sub> model and then we calculate the distribution p(x) by running Mp on prefix while in parallel speculatively calculating the distribution of the next token x2 by running Mp on prefix + [x1]. Once both computations complete we proceed further same as above.
- If x1 is rejected we discard the computation x2 and resample from x1 from an adjusted distribution if x1 is accepted we keep both tokens

### Algorithm
Inputs: M<sub>p</sub>, M<sub>q</sub>
sample Y guesses x1,...y from Mq autoregrssively  
```
for i = 1 to Y: 
we do   
qi(x) <-- Mq( prefix + [ x1, ...xi-1] )  
xi ~ qi(x)
end for 
Run Mp in parallel
p1(x)...py+1(x) <-- Mp(prefix)....Mp(prefix + [x1,....xy])
determine the number of accepted guesses n
r1 ~ U(0, 1),..., ry ~ U(0,1)
n <-- min({i-1 | 1<= i <= y, ri > pi(x)/qi(x)}U{Y})
adjust the distribution from Mp if needed   
p'(x) <- norm(max(0, pn+1(x) - qn+1(x)))
end if
return one token from Mp, and n tokens from Mq
retrun prefix + [x1, ..., xn, t]
```
