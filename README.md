# Linear-Time-Construction-of-Suffix-Arrays
## The Skew Algorithm (aka DC3)

This project is a working Python3 implementation of the Skew algorithm, also known as the DC3 algorithm that I wrote in the Jupyter Notebook environment . It constructs suffix arrays in linear-time, making it particularly suited for applications in the field of bioinformatics.

## Algorithm Outline

1. The suffixes that do not start at positions that are multiples of 3 (i.e., positions i mod 3 ≠ 0) are stored in SA12. 
2. The suffixes starting at positions that are multiples of 3 are stored in SA3. These suffixes are then sorted using bucket sort.
3. Merge the sorted suffix array SA12 with the sorted suffix array SA3.

## Example
Construct the suffix array of a string.
- Input: A string Text.
- Output: SuffixArray(Text).

**Sample Input:**
AACGATAGCGGTAGA$

**Sample Output:**
15, 14, 0, 1, 12, 6, 4, 2, 8, 13, 3, 7, 9, 10, 11, 5

## Usage
To use the Skew algorithm, follow these steps:

**Installation**
1. Clone this repo:
```
git clone https://github.com/jvarg122/Linear-Time-Construction-of-Suffix-Arrays.git
```
2. Dependencies:

```
pip install matplotlib
```
Install Matplotlib from https://matplotlib.org/stable/

**Run the script**
```
python "suffix_array.py"
```
[Optional] Alternatively, you may download the `suffix_array.ipynb` file and execute the notebook by selecting Run -> Run All Cells in [Juypter](https://jupyter.org/).

A snapshot of the code in operation demonstrating how the results should appear. 

<img src= "https://i.imgur.com/8zUQABY.png">

## Collected Data
The table shows the execution times (in seconds) of different algorithms (Naive, Fast, and Skew) for varying input sizes.

<img src= "https://i.imgur.com/2howfkk.png">

Data collected during the execution of the algorithm is visualized below:

<img src= "https://i.imgur.com/FHUw8yF.png">

<img src= "https://i.imgur.com/uNwxcbR.png">

<img src= "https://i.imgur.com/UgjnD9r.png">

<img src= "https://i.imgur.com/Klq8G2T.png">

## References
- [Research paper where the algorithm was initally introduced](https://www.cs.cmu.edu/~guyb/paralg/papers/KarkkainenSanders03.pdf) by Juha K¨arkk¨ainen and Peter Sanders
- [Louis Abraham's](https://louisabraham.github.io/articles/suffix-arrays) Suffix arrays in Python 










