# Learn parity of bit strings with LSTM
Solution to openAI's [Request for Research 2.0](https://openai.com/blog/requests-for-research-2/) warm-up question with less than 90 lines of codes.

# The Question
Paraphrase the question here since the page appeared not maintained. Not sure if it will go away.
>Train an LSTM to solve the XOR problem: that is, given a sequence of bits, determine its parity. The LSTM should consume the sequence, one bit at a time, and then output the correct answer at the sequence’s end. Test the two approaches below:

>Generate a dataset of random 100,000 binary strings of length 50. Train the LSTM; what performance do you get?

>Generate a dataset of random 100,000 binary strings, where the length of each string is independently and randomly chosen between 1 and 50. Train the LSTM. Does it succeed? What explains the difference?

A binary string of 50 bits has $2^50 ~ 10^15$ different sequences. Any netowrk won't be able to just memorize the answer using 100,000 examples. In the codes I used another 1,000 sequences for validation. There's no chance those sequences would be the same as the training sequences. In other words, to solve this problem the network has to learn to calculate parity from an arbitrary binary sequence.

# Result
## Variable length
Both fixed length and variable length training samples works. It takes more variable length training samples to acheive 100% accuracy.
* Fixed length:    60,000 samples
* Variable length: 150,000 samples

## Generalize to longer sequence length
The model still works when training with 50-bit samples and predict the parity of 100-bit sequence. The same number of samples (60,000) are needed. The model is able to generalize to the sequence length it was not trained on.

## Batch size
Small batch size works better in training.

|Batch size | Samples needed       |
|-----------|----------------------|
|8          |  60,000              |
|16         |  80,000              |
|32         | 160,256              |
|64         | 311,392              |

## Number of layers of LSTM
Shallow network is easier to train and does the job equally well. Deeper is not always better.

|Layers     | Samples needed (SGD) | Samples needed (ADAM) |
|-----------|----------------------|-----------------------|
|1          |  60,000              | 10,000                |
|2          |  150,000             | 200,000               |
|3          |  500,000             | Not converged         |

## Number of hidden units

|Hidden units | Samples needed       |
|-------------|----------------------|
|2            |  Not converged       |
|4            |  10,000              |
|8            |  10,000              |
|16           |  10,000              |
|32           |  10,000              |
|64           |  Not converged       |
(1 layer, ADAM optimizer)

# Conclusion
For this problem, it's the best to use LSTM with 1 layer and 4-32 hidden units.

It's also important to formulate the training well. The following sample, target pair does not work well:

```
sample: 011001
target: 1   (parity of the string)
```

Instead this does:

```
sample: 011001
target: 010001 (parity up to the bit position)
```

The reason is the latter also gives target to intermediate strings (with lengths 1, 2, 3, etc), not just the final one. Thus casting the training this way provides a lot more direction to how the network should behave.
