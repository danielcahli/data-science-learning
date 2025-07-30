#NumPy stands for Numerical Python.
#NumPy aims to provide an array object that is up to 50x faster than traditional Python lists.
#NumPy arrays are stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently.
#Data Science: is a branch of computer science where we study how to store, use and analyze data for deriving information from it.
#The source code for NumPy is located at this github repository https://github.com/numpy/numpy

import numpy as np
from numpy import random

print(np.__version__)
#2.2.0

arr = np.array([1, 2, 3, 4, 5])

print(arr)

print(type(arr))
#<class 'numpy.ndarray'>


#Create a 3-D array with two 2-D arrays, both containing two arrays with the values 1,2,3 and 4,5,6:

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr)

#Check how many dimensions the arrays have:

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

#Create an array with 5 dimensions and verify that it has 5 dimensions:

arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('number of dimensions :', arr.ndim) 

#Get the first element from the following array:

arr = np.array([1, 2, 3, 4])

print(arr[0]) 

#Get third and fourth elements from the following array and add them.

arr = np.array([1, 2, 3, 4])

print(arr[2] + arr[3]) 

#To access elements from 2-D arrays we can use comma separated integers representing the dimension and the index of the element.

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('2nd element on 1st row: ', arr[0, 1])

#To access elements from 3-D arrays we can use comma separated integers representing the dimensions and the index of the element.

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr[0, 1, 2]) 

# Use negative indexing to access an array from the end.

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('Last element from 2nd dim: ', arr[1, -1])

#Slice elements from index 1 to index 5 from the following array:

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5])

#Slice from the index 3 from the end to index 1 from the end:

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[-3:-1])

#Return every other element from index 1 to index 5:

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5:2]) 

#Return every other element from the entire array:

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[::2]) 

#From the second element, slice elements from index 1 to index 4 (not included):

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[1, 1:4]) # [7 8 9] 

#From both elements, return index 2:

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[0:2, 2]) # [3 8]

#Get the data type of an array object:

arr = np.array([1, 2, 3, 4])

print(arr.dtype) #int64 

arr = np.array(['apple', 'banana', 'cherry'])

print(arr.dtype) #U6 

#Create an array with data type string:
arr = np.array([1, 2, 3, 4], dtype='S')

print(arr) #[b'1' b'2' b'3' b'4']
print(arr.dtype) #|S1 

#Create an array with data type 4 bytes integer:

arr = np.array([1, 2, 3, 4], dtype='i4')

print(arr) #[1 2 3 4]
print(arr.dtype) #int32 

#A non integer string like 'a' can not be converted to integer (will raise an error):

arr = np.array(['a', '2', '3'], dtype='i') #ValueError: invalid literal for int() with base 10: 'a' 

#Change data type from float to integer by using 'i' as parameter value:

arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype('i')

print(newarr) #[1 2 3]
print(newarr.dtype) #int32 

#Change data type from float to integer by using int as parameter value:

arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype(int)

print(newarr)
print(newarr.dtype) #int64 

#Change data type from integer to boolean:

arr = np.array([1, 0, 3])

newarr = arr.astype(bool)

print(newarr) #[ True False  True]
print(newarr.dtype) #bool 

#Make a copy, change the original array, and display both arrays:

arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42

print(arr) #[42  2  3  4  5]
print(x) #[1 2 3 4 5] 

#Make a view, change the view, and display both arrays:

arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
x[0] = 31

print(arr) #[31  2  3  4  5]
print(x) #[31  2  3  4  5] 

#Print the value of the base attribute to check if an array owns it's data or not:

arr = np.array([1, 2, 3, 4, 5])

x = arr.copy()
y = arr.view()

print(x.base) #None
print(y.base) #[1 2 3 4 5] 

#Print the shape of a 2-D array:

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(arr.shape) #(2, 4)

#Create an array with 5 dimensions using ndmin using a vector with values 1,2,3,4 and verify that last dimension has value 4:

arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr) #[[[[[1 2 3 4]]]]]
print('shape of array :', arr.shape) #shape of array : (1, 1, 1, 1, 4) 

#Convert the following 1-D array with 12 elements into a 2-D array.

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)

print(newarr)
#[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]] 

#Convert the following 1-D array with 12 elements into a 3-D array.

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(2, 3, 2)

print(newarr)
[[[ 1  2]
  [ 3  4]
  [ 5  6]]

 [[ 7  8]
  [ 9 10]
  [11 12]]]

#Try converting 1D array with 8 elements to a 2D array with 3 elements in each dimension (will raise an error): 

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

newarr = arr.reshape(3, 3)

print(newarr) #ValueError: cannot reshape array of size 8 into shape (3,3) 

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print(arr.reshape(2, 4).base) #[1 2 3 4 5 6 7 8] #OBS: ITS A VIEW, BECOUSE RETURN THE ORIGINAL

#Convert 1D array with 8 elements to 3D array with 2x2 elements:

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

newarr = arr.reshape(2, 2, -1)

print(newarr)  
#[[[1 2]
#  [3 4]]

# [[5 6]
#  [7 8]]]
obs: You are allowed to have one "unknown" dimension. 

#Flattening array means converting a multidimensional array into a 1D array.

#We can use reshape(-1) to do this.

arr = np.array([[1, 2, 3], [4, 5, 6]])

newarr = arr.reshape(-1)

print(newarr)
# [1 2 3 4 5 6] 

#Iterate on the elements of the following 1-D array:

arr = np.array([1, 2, 3])

for x in arr:
  print(x)
#1
#2
#3 

#Iterate on the elements of the following 2-D array:

arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  print(x)
#[1 2 3]
#[4 5 6]

#Iterate on each scalar element of the 2-D array:

arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  for y in x:
    print(y) 
#1
#2
#3
#4
#5
#6

#Iterate on the elements of the following 3-D array:

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  print(x)
 # [[1 2 3]
 #[4 5 6]]
#[[ 7  8  9]
 #[10 11 12]] 

#To return the actual values, the scalars, we have to iterate the arrays in each dimension.

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  for y in x:
    for z in y:
      print(z, end=' ')
# 1 2 3 4 5 6 7 8 9 10 11 12 

arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for x in np.nditer(arr):
  print(x, end=' ')
# 1 2 3 4 5 6 7 8 

#Iterate through the array as a string:
#NumPy does not change the data type of the element in-place (where the element is in array) so it needs some other space to perform this action, that extra space is called buffer, and in order to enable it in nditer() we pass flags=['buffered'].

arr = np.array([1, 2, 3])

for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x)
#np.bytes_(b'1')
#np.bytes_(b'2')
#np.bytes_(b'3')  

#Iterate through every scalar element of the 2D array skipping 1 element:

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for x in np.nditer(arr[:, ::2]):
  print(x, end='')
#1357  

#Sometimes we require corresponding index of the element while iterating,

arr = np.array([1, 2, 3])

for idx, x in np.ndenumerate(arr):
  print(idx, x, end = " ")
# (0,) 1 (1,) 2 (2,) 3 

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) 

for idx, x in np.ndenumerate(arr):
  print(idx, x, end = " - ")
#(0, 0) 1 - (0, 1) 2 - (0, 2) 3 - (0, 3) 4 - (1, 0) 5 - (1, 1) 6 - (1, 2) 7 - (1, 3) 8 - 

#In SQL we join tables based on a key, whereas in NumPy we join arrays by axes.
#We pass a sequence of arrays that we want to join to the concatenate() function, along with the axis. If axis is not explicitly passed, it is taken as 0. 

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.concatenate((arr1, arr2)) 

print(arr)
#[1 2 3 4 5 6]

arr1 = np.array([[1, 2], [3, 4]])

arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis=1)

print(arr)
#[[1 2 5 6]
# [3 4 7 8]] 

#Stacking is same as concatenation, the only difference is that stacking is done along a new axis.

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print(arr)
#[[1 4]
 #[2 5]
 #[3 6]]  

#To stack along rows:

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.hstack((arr1, arr2))

print(arr)

#[1 2 3 4 5 6]

#To stack along columns: 

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.vstack((arr1, arr2))

print(arr)
#[[1 2 3]
 #[4 5 6]] 

#To stack along height, which is the same as depth:
arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.dstack((arr1, arr2))

print(arr)
#[[[1 4]
 #[2 5]
 #[3 6]]] 

#Split the array in 3 parts:

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr)
#[array([1, 2]), array([3, 4]), array([5, 6])] 

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 4)

print(newarr)
#[array([1, 2]), array([3, 4]), array([5]), array([6])] 

#Note: We also have the method split() available but it will not adjust the elements when elements are less in source array for splitting like in example above, array_split() worked properly but split() would fail.

#The return value of the array_split() method is an array containing each of the split as an array.  

arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr[0])
print(newarr[1])
print(newarr[2])
#[1 2]
#[3 4]
#[5 6] 

#Split the 2-D array into three 2-D arrays.

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.array_split(arr, 3)
print(newarr)

#[array([[1, 2, 3],
       #[4, 5, 6]]), array([[ 7,  8,  9],
       #[10, 11, 12]]), array([[13, 14, 15],
       #[16, 17, 18]])]
#The example below also returns three 2-D arrays, but they are split along the column (axis=1).  

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.array_split(arr, 3, axis=1) 

print(newarr)
#[array([[ 1],
       #[ 4],
       #[ 7],
       #[10],
       #[13],
       #[16]]), array([[ 2],
       #[ 5],
       #[ 8],
       #[11],
       #[14],
       #[17]]), array([[ 3],
       #[ 6],
       #[ 9],
       #[12],
       #[15],
       #[18]])] 

An alternate solution is using hsplit() opposite of hstack() 

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.hsplit(arr, 3)

print(newarr)
'''[array([[ 1],
       [ 4],
       [ 7],
       [10],
       [13],
       [16]]), array([[ 2],
       [ 5],
       [ 8],
       [11],
       [14],
       [17]]), array([[ 3],
       [ 6],
       [ 9],
       [12],
       [15],
       [18]])]''' 

#Note: Similar alternates to vstack() and dstack() are available as vsplit() and dsplit(). 

#You can search an array for a certain value, and return the indexes that get a match. 

arr = np.array([1, 2, 3, 4, 5, 4, 4])

x = np.where(arr == 4)

print(x)
#(array([3, 5, 6]),) 

#Find the indexes where the values are even: 

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x = np.where(arr%2 == 0)

print(x)
#(array([1, 3, 5, 7]),) 

#Find the indexes where the values are odd:

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

x = np.where(arr%2 == 1)

print(x)
#(array([0, 2, 4, 6]),) 

#There is a method called searchsorted() which performs a binary search in the array, and returns the index where the specified value would be inserted to maintain the search order.
#The searchsorted() method is assumed to be used on sorted arrays.
arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 7)

print(x) #1

#By default the left most index is returned, but we can give side='right' to return the right most index instead.  

arr = np.array([6, 7, 8, 9])

x = np.searchsorted(arr, 7, side='right')

print(x) #2  

arr = np.array([1, 3, 5, 7])

x = np.searchsorted(arr, [2, 4, 6])

print(x) #[1 2 3] 

#Sort the array:

arr = np.array([3, 2, 0, 1])

print(np.sort(arr)) #[0 1 2 3]

#Sort the array alphabetically: 

arr = np.array(['banana', 'cherry', 'apple'])

print(np.sort(arr)) #['apple' 'banana' 'cherry'] 

#If you use the sort() method on a 2-D array, both arrays will be sorted:

arr = np.array([[3, 2, 4], [5, 0, 1]])

print(np.sort(arr))
'''[[2 3 4]
 [0 1 5]]''' 

'''Getting some elements out of an existing array and creating a new array out of them is called filtering.

In NumPy, you filter an array using a boolean index list. ''' 

arr = np.array([41, 42, 43, 44])

x = [True, False, True, False]

newarr = arr[x]

print(newarr) #[41 43]
 
#Create a filter array that will return only values higher than 42:

arr = np.array([41, 42, 43, 44])

filter_arr = []

for element in arr:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)
#[False, False, True, True]
#[43 44] 

#We can directly substitute the array instead of the iterable variable in our condition and it will work just as we expect it to.

arr = np.array([41, 42, 43, 44])

filter_arr = arr > 42

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)
#[False False  True  True]
#[43 44]

#Create a filter array that will return only even elements from the original array:

arr = np.array([1, 2, 3, 4, 5, 6, 7])

filter_arr = arr % 2 == 0

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)
#[False  True False  True False  True False]
#[2 4 6]

'''If there is a program to generate random number it can be predicted, thus it is not truly random.
Random numbers generated through a generation algorithm are called pseudo random. 
In order to generate a truly random number on our computers we need to get the random data from some outside source. This outside source is generally our keystrokes, mouse movements, data on network etc.'''

#Generate a random integer from 0 to 100:

x = random.randint(100)

print(x) #67

#The random module's rand() method returns a random float between 0 and 1.

x = random.rand()

print(x) #0.37983444207121797

#The randint() method takes a size parameter where you can specify the shape of an array.

x=random.randint(100, size=(5))

print(x) #[51 29 75 98  5]

#Generate a 2-D array with 3 rows, each row containing 5 random integers from 0 to 100:

x = random.randint(100, size=(3, 5))

print(x)
'''
[[98 73 10 77 27]
 [36 98 41 60 11]
 [99 51 57 10 67]]'''

#The rand() method also allows you to specify the shape of the array.

x = random.rand(5)

print(x)
#[0.4326867  0.01742127 0.45125027 0.57031187 0.69900429]

x = random.rand(3, 5)
print(x)'''
[[0.83955853 0.03601323 0.28534932 0.21047603 0.0677224 ]
 [0.19079739 0.17732727 0.00632526 0.1656446  0.97562732]
 [0.08925391 0.88600071 0.36014353 0.52327994 0.51259694]]'''

#The choice() method takes an array as a parameter and randomly returns one of the values.
x = random.choice([3, 5, 7, 9])

print(x) #9

x = random.choice([3, 5, 7, 9], size=(3, 5))

print(x) '''
[[3 7 9 3 9]
 [3 9 9 7 3]
 [9 5 5 9 3]]'''

#The probability is set by a number between 0 and 1, where 0 means that the value will never occur and 1 means that the value will always occur.

x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))

print(x)
''' [7 7 7 5 7 3 7 7 5 7 7 7 7 3 7 7 7 7 5 7 5 7 7 7 5 5 7 7 5 7 7 5 7 7 5 5 7
 3 5 3 5 5 5 3 7 5 7 7 7 7 7 5 5 7 5 7 7 7 7 7 5 5 7 7 7 3 7 3 7 5 7 5 3 7
 7 5 7 5 5 5 7 7 3 7 5 7 7 5 7 5 7 7 7 7 5 3 7 7 7 5]'''
 
#The shuffle() method makes changes to the original array.
#The permutation() method returns a re-arranged array (and leaves the original array un-changed).

arr = np.array([1, 2, 3, 4, 5])

random.shuffle(arr)

print(arr) #[4 3 5 1 2]

arr = np.array([1, 2, 3, 4, 5])

print(random.permutation(arr)) #[4 3 1 2 5]

r'''Seaborn is a library that uses Matplotlib underneath to plot graphs. It will be used to visualize random distributions.
Install Seaborn.
If you have Python and PIP already installed on a system, install it using this command:
C:/Users/Your Name>pip install seaborn
If you use Jupyter, install Seaborn using this command:
C:/Users/Your Name>!pip install seaborn
Displot stands for distribution plot, it takes as input an array and plots a curve corresponding to the distribution of points in the array.
Import the pyplot object of the Matplotlib module in your code using the following statement:
Import the Seaborn module in your code using the following statement:'''
"""
import matplotlib.pyplot as plt
import seaborn as sns
'''sns.displot([0, 1, 2, 3, 4, 5])

plt.show()# é gerado um histograma

#Plotting a Displot Without the Histogram

sns.displot([0, 1, 2, 3, 4, 5], kind="kde")

plt.show() #é gerada uma curva de distribuição

The Normal Distribution is also called the Gaussian Distribution 
Use the random.normal() method to get a Normal Data Distribution.
It has three parameters:
loc - (Mean) where the peak of the bell exists.
scale - (Standard Deviation) how flat the graph distribution should be.
size - The shape of the returned array. 
x = random.normal(size=(2, 3))
print(x)
#[[ 0.6574879  -1.3580887  -1.41663631]
 #[-0.31657066  1.06789481  0.47955757]]

x = random.normal(loc=1, scale=2, size=(2, 3))
print(x)
#[[ 4.27596898 -0.78431529  0.71222421]
# [ 4.83871332 -2.77358976  1.05590358]]
 
sns.displot(random.normal(size=1000), kind="kde")

plt.show() #visualisação da curva normal (bell curve).

Binomial Distribution is a Discrete Distribution.
It describes the outcome of binary scenarios, e.g. toss of a coin, it will either be head or tails.
It has three parameters:
n - number of trials.
p - probability of occurence of each trial.
size - The shape of the returned array.

x = random.binomial(n=10, p=0.5, size=10)

print(x) #[6 6 4 5 8 5 6 8 1 7]

sns.displot(random.binomial(n=10, p=0.5, size=1000))

plt.show() # mostra um histograma

data = {
  "normal": random.normal(loc=50, scale=5, size=1000),
  "binomial": random.binomial(n=100, p=0.5, size=1000)
}

sns.displot(data, kind="kde")

plt.show() #mostra uma curva normal e uma binomial quase coincidentes

#Poisson Distribution is a Discrete Distribution.
#It estimates how many times an event can happen in a specified time. e.g. If someone eats twice a day what is the probability he will eat thrice?
#It has two parameters:
#lam - rate or known number of occurrences e.g. 2 for above problem.
#size - The shape of the returned array.

x = random.poisson(lam=2, size=10)

print(x) [3 3 0 3 0 4 1 2 6 1]

sns.displot(random.poisson(lam=2, size=1000))

plt.show() #exibe um histograma

data = {
  "normal": random.normal(loc=50, scale=7, size=1000),
  "poisson": random.poisson(lam=50, size=1000)
}

sns.displot(data, kind="kde")

plt.show() #exibe uma curva normal e uma poisson parecidos

#Normal distribution is continuous whereas poisson is discrete.
#Binomial distribution only has two possible outcomes, whereas poisson distribution can have unlimited possible outcomes.
#But for very large n and near-zero p binomial distribution is near identical to poisson distribution such that n * p is nearly
#equal to lam.

data = {
  "binomial": random.binomial(n=1000, p=0.01, size=1000),
  "poisson": random.poisson(lam=10, size=1000)
}

sns.displot(data, kind="kde")

plt.show() #exibe uma curva binomial e uma poisson parecidos

#Uniform Distribution
#Used to describe probability where every event has equal chances of occuring.
#E.g. Generation of random numbers.
#It has three parameters:
#low - lower bound - default 0 .0.
#high - upper bound - default 1.0.
#size - The shape of the returned array.

x = random.uniform(size=(2, 3))

print(x)
#[[0.19047991 0.1405641  0.97907813]
# [0.73878001 0.72219172 0.34191538]]

sns.displot(random.uniform(size=1000), kind="kde")

plt.show() #grafico da distribuição

#obs: O KDE transforma os seus dados discretos em uma curva contínua que estima a densidade de probabilidade. Essa curva mostra onde os dados estão mais concentrados, suavizando flutuações bruscas.

#Logistic Distribution is used to describe growth. Used extensively in machine learning in logistic regression, neural networks etc. It has three parameters:
#loc - mean, where the peak is. Default 0.
#scale - standard deviation, the flatness of distribution. Default 1.
#size - The shape of the returned array

#Draw 2x3 samples from a logistic distribution with mean at 1 and stddev 2.0:

x = random.logistic(loc=1, scale=2, size=(2, 3))

print(x)
#[[ 4.04014271 -2.08989808 14.09057412]
 #[ 0.95141817 -1.84025465 -9.65378424]]

#Visualization of Logistic Distribution

sns.displot(random.logistic(size=1000), kind="kde")

plt.show()

#Difference Between Logistic and Normal Distribution
#Both distributions are near identical, but logistic distribution has more area under the tails, meaning it represents more possibility of occurrence of an event further away from mean.
#For higher value of scale (standard deviation) the normal and logistic distributions are near identical apart from the peak.

data = {
  "normal": random.normal(scale=2, size=1000),
  "logistic": random.logistic(size=1000)
}

sns.displot(data, kind="kde")

plt.show()

#Multinomial distribution is a generalization of binomial distribution.

#It describes outcomes of multi-nomial scenarios unlike binomial where scenarios must be only one of two.

#It has three parameters: n - number of times to run the experiment. 
#pvals - list of probabilties of outcomes
#size - The shape of the returned array.

#Draw out a sample for dice roll: 

x = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

print(x) #[1 0 0 0 2 3] 

#Note: As they are generalization of binomial distribution their visual representation and similarity of normal distribution is same as that of multiple binomial distributions.

#Exponential distribution is used for describing time till next event e.g. failure/success etc.
#It has two parameters:
#scale - inverse of rate ( see lam in poisson distribution ) defaults to 1.0.
#size - The shape of the returned array.
#EX:
x = random.exponential(scale=2, size=(2, 3))

print(x)

#Visualization:

sns.displot(random.exponential(size=1000), kind="kde")

plt.show()
#Poisson distribution deals with number of occurences of an event in a time period whereas exponential distribution deals with the time between these events.

#Chi Square distribution is used as a basis to verify the hypothesis.

#It has two parameters: df - (degree of freedom). size - The shape of the returned array.

#Ex: Draw out a sample for chi squared distribution with degree of freedom 2 with size 2x3:

from numpy import random

x = random.chisquare(df=2, size=(2, 3))
print(x)
#[[ 0.05528149  0.1952845   1.17161855]
 #[ 1.10886367  1.17385769 10.60044082]]

#Example: visualization of Chi Square Distribution

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.displot(random.chisquare(df=1, size=1000), kind="kde")

plt.show()

#Rayleigh distribution is used in signal processing.
#It has two parameters: scale - (standard deviation) decides how flat the distribution will be default 1.0.
#size - The shape of the returned array.
#Example: Draw out a sample for rayleigh distribution with scale of 2 with size 2x3:

x = random.rayleigh(scale=2, size=(2, 3))

print(x)

#Visualization of Rayleigh Distribution. Example:

sns.displot(random.rayleigh(size=1000), kind="kde")

plt.show()

# OBS: At unit stddev and 2 degrees of freedom rayleigh and chi square represent the same distributions.

#A distribution following Pareto's law i.e. 80-20 distribution (20% factors cause 80% outcome).

#It has two parameter: a - shape parameter. size - The shape of the returned array.

#Draw out a sample for pareto distribution with shape of 2 with size 2x3:

x = random.pareto(a=2, size=(2, 3))

print(x)
#[[0.33974085 0.79765637 0.23923772]
 #[0.84370981 0.07726483 0.49431438]]

#Visualization of Pareto Distribution

sns.displot(random.pareto(a=2, size=1000))

plt.show() 

#Zipf distributions are used to sample data based on zipf's law.
#Zipf's Law: In a collection, the nth common term is 1/n times of the most common term. E.g. the 5th most common word in English occurs nearly 1/5 times as often as the most common word.
#It has two parameters: a - distribution parameter. size - The shape of the returned array.

#Draw out a sample for zipf distribution with distribution parameter 2 with size 2x3:

x = random.zipf(a=2, size=(2, 3))

print(x)
#[[  1   1 139]
 #[  3   3   1]]

#Visualization of Zipf Distribution
#Sample 1000 points but plotting only ones with value < 10 for more meaningful chart.

x = random.zipf(a=2, size=1000)
sns.displot(x[x<20])

plt.show()

#ufuncs stands for "Universal Functions" and they are NumPy functions that operate on the ndarray object.
#ufuncs are used to implement vectorization in NumPy which is way faster than iterating over elements.
#They also provide broadcasting and additional methods like reduce, accumulate etc. that are very helpful for computation.
#ufuncs also take additional arguments, like:
#where boolean array or condition defining where the operations should take place.
#dtype defining the return type of elements.
#out output array where the return value should be copied.
#Converting iterative statements into a vector based operation is called vectorization.
#It is faster as modern CPUs are optimized for such operations.

#Add the Elements of Two Lists
#list 1: [1, 2, 3, 4]
#list 2: [4, 5, 6, 7]
#One way of doing it is to iterate over both of the lists and then sum each elements.
#Without ufunc, we can use Python's built-in zip() method:

x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = []

for i, j in zip(x, y):
  z.append(i + j)
print(z)
#[5, 7, 9, 11]
#NumPy has a ufunc for this, called add(x, y) that will produce the same result.

x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = np.add(x, y)
print(z)
#[ 5  7  9 11] 

#To create your own ufunc, you have to define a function, like you do with normal functions in Python, then you add it to your NumPy ufunc library with the frompyfunc() method.
#The frompyfunc() method takes the following arguments:
#function - the name of the function.
#inputs - the number of input arguments (arrays).
#outputs - the number of output arrays.
#Example: Create your own ufunc for addition:

def myadd(x, y):
  return x+y

myadd = np.frompyfunc(myadd, 2, 1)

print(myadd([1, 2, 3, 4], [5, 6, 7, 8]))
#[6 8 10 12]

#Check if a Function is a ufunc

print(type(np.add)) #<class 'numpy.ufunc'>

print(type(np.concatenate)) #<class 'numpy._ArrayFunctionDispatcher'>

print(type(np.blahblah)) #AttributeError: module 'numpy' has no attribute 'blahblah'

#To test if the function is a ufunc in an if statement, use the numpy.ufunc value (or np.ufunc if you use np as an alias for numpy): 

if type(np.add) == np.ufunc:
  print('add is ufunc')
else:
  print('add is not ufunc')
#add is ufunc 

#Simple Arithmetic
#You could use arithmetic operators + - * / directly between NumPy arrays, but this section discusses an extension of the same where we have functions that can take any array-like objects e.g. lists, tuples etc. and perform arithmetic conditionally.
#Arithmetic Conditionally: means that we can define conditions where the arithmetic operation should happen.
#All of the discussed arithmetic functions take a where parameter in which we can specify that condition.

arr1 = np.array([10, 11, 12, 13, 14, 15])
arr2 = np.array([1, 2, 2, 3, 4, 5])

newarr = np.add(arr1, arr2)

print(newarr) #[11 13 14 16 18 20]

newarr = np.subtract(arr1, arr2) 

print(newarr) #[ 9  9 10 10 10 10]

newarr = np.multiply(arr1, arr2)

print(newarr) #[10 22 24 39 56 75]

newarr = np.divide(arr1, arr2)

print(newarr) #[10.          5.5         6.          4.33333333  3.5         3.        ]

newarr = np.power(arr1, arr2)

print(newarr) #[    10    121    144   2197  38416 759375]
#Both the mod() and the remainder() functions return the remainder of the values in the first array corresponding to the values in the second array, and return the results in a new array.

newarr = np.mod(arr1, arr2)

print(newarr) #[0 1 0 1 2 0]

newarr = np.remainder(arr1, arr2)

print(newarr) #[0 1 0 1 2 0]

#The divmod() function return both the quotient and the mod. The return value is two arrays, the first array contains the quotient and second array contains the mod.

newarr = np.divmod(arr1, arr2) #(array([10,  5,  6,  4,  3,  3]), array([0, 1, 0, 1, 2, 0]))

print(newarr)

newarr = np.absolute(arr1) 

print(newarr) #[10 11 12 13 14 15] 

#There are primarily five ways of rounding off decimals in NumPy:

#truncation
arr = np.trunc([-3.1666, 3.6667])

print(arr) #[-3.  3.]

arr = np.fix([-3.1666, 3.6667])

print(arr) # [-3.  3.]

#Rounding

arr = np.around(3.1666, 2)

print(arr) #3.17

#Floor

arr = np.floor([-3.1666, 3.6667])

print(arr) #[-4.  3.]

#Ceil

arr = np.ceil([-3.1666, 3.6667])

print(arr) #[-3.  4.]

#Log at Base 2
#Note: The arange(1, 10) function returns an array with integers starting from 1 (included) to 10 (not included).

arr = np.arange(1, 10)

print(np.log2(arr))
#[0.         1.         1.5849625  2.         2.32192809 2.5849625
 #2.80735492 3.         3.169925  ]

#Log at Base 10

print(np.log10(arr))
#[0.         0.30103    0.47712125 0.60205999 0.69897    0.77815125
# 0.84509804 0.90308999 0.95424251]

#Natural Log, or Log at Base e

print(np.log(arr))
#[0.         0.69314718 1.09861229 1.38629436 1.60943791 1.79175947
 #1.94591015 2.07944154 2.19722458]

#Log at Any Base
#NumPy does not provide any function to take log at any base, so we can use the frompyfunc() function along with inbuilt function math.log() with two input parameters and one output parameter:

from math import log
import numpy as np

nplog = np.frompyfunc(log, 2, 1)

print(nplog(100, 15))
#1.7005483074552052 

#Addition is done between two arguments whereas summation happens over n elements.

arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])

newarr = np.add(arr1, arr2)

print(newarr) #[2 4 6]

newarr = np.sum([arr1, arr2])

print(newarr) #12

#If you specify axis=1, NumPy will sum the numbers in each array.

newarr = np.sum([arr1, arr2], axis=1)

print(newarr) #[6 6]

#Cummulative sum means partially adding the elements in array.

newarr = np.cumsum(arr1)

print(newarr) #[1 3 6] 

#Find the product of the elements of this array:

arr = np.array([1, 2, 3, 4])

x = np.prod(arr)

print(x) #24

#Find the product of the elements of two arrays:

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

x = np.prod([arr1, arr2])

print(x) #40320

#If you specify axis=1, NumPy will return the product of each array.

newarr = np.prod([arr1, arr2], axis=1)

print(newarr) #[  24 1680]

#Perfom partial sum with the cumprod() function.

newarr = np.cumprod(arr)

print(newarr) #[ 1  2  6 24] 


#A discrete difference means subtracting two successive elements.

arr = np.array([10, 15, 25, 5])

newarr = np.diff(arr)

print(newarr)
# Returns: [5 10 -20] because 15-10=5, 25-15=10, and 5-25=-20

newarr = np.diff(arr, n=2)

print(newarr)
#Returns: [5 -30] because: 15-10=5, 25-15=10, and 5-25=-20 AND 10-5=5 and -20-10=-30

#Find the LCM of the following two numbers:

num1 = 4
num2 = 6

x = np.lcm(num1, num2)

print(x)
#Returns: 12 because that is the lowest common multiple of both numbers (4*3=12 and 6*2=12).

#The reduce() method will use the ufunc, in this case the lcm() function, on each element, and reduce the array by one dimension.

arr = np.array([3, 6, 9])

x = np.lcm.reduce(arr)

print(x)
#Returns: 18 because that is the lowest common multiple of all three numbers (3*6=18, 6*3=18 and 9*2=18).

#Find the LCM of all values of an array where the array contains all integers from 1 to 10:

arr = np.arange(1, 11)

x = np.lcm.reduce(arr)

print(x) #2520 

#The GCD (Greatest Common Devisor), also known as HCF (Highest Common Factor) is the biggest number that is a common factor of both of the numbers.

num1 = 6
num2 = 9

x = np.gcd(num1, num2)

print(x) #3

arr = np.array([20, 8, 32, 36, 16])

x = np.gcd.reduce(arr)

print(x) #4

#Find sine values: 

x = np.sin(np.pi/2)

print(x) #1.0

arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])

x = np.sin(arr)

print(x) #[1.         0.8660254  0.70710678 0.58778525]

#Convert all of the values in following array arr to radians: 

arr = np.array([90, 180, 270, 360])

x = np.deg2rad(arr)

print(x) #[1.57079633 3.14159265 4.71238898 6.28318531]

#Convert all of the values in following array arr to degrees:

arr = np.array([np.pi/2, np.pi, 1.5*np.pi, 2*np.pi])

x = np.rad2deg(arr)

print(x) #[ 90. 180. 270. 360.] 

#Find the angle of 1.0:

x = np.arcsin(1.0)

print(x) #1.5707963267948966

#Find the angle for all of the sine values in the array 

arr = np.array([1, -1, 0.1])

x = np.arcsin(arr)

print(x) #[ 1.57079633 -1.57079633  0.10016742]

#Find the hypotenues for 4 base and 3 perpendicular: 

base = 3
perp = 4

x = np.hypot(base, perp)

print(x) #5.0

#Find sinh value of PI/2: 

x = np.sinh(np.pi/2)

print(x) #2.3012989023072947

#Find cosh values for all of the values in arr: 


arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])

x = np.cosh(arr)

print(x) #[2.50917848 1.60028686 1.32460909 1.20397209] 

#Find the angle of 1.0:

x = np.arcsinh(1.0)

print(x) #0.881373587019543 

#Find the angle for all of the tanh values in array:

arr = np.array([0.1, 0.2, 0.5])

x = np.arctanh(arr)

print(x) #[0.10033535 0.20273255 0.54930614] 

#Convert following array with repeated elements to a set:

arr = np.array([1, 1, 1, 2, 3, 4, 5, 5, 6, 7])

x = np.unique(arr)

print(x) #[1 2 3 4 5 6 7] 

#To find the unique values of two arrays, use the union1d() method.

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])

newarr = np.union1d(arr1, arr2)

print(newarr) #[1 2 3 4 5 6] 

#Find intersection of the following two set arrays:

newarr = np.intersect1d(arr1, arr2, assume_unique=True)

print(newarr) #[3 4]

#To find only the values in the first set that is NOT present in the seconds set, use the setdiff1d() method.

#Example: Find the difference of the set1 from set2:

newarr = np.setdiff1d(arr1, arr2, assume_unique=True)

print(newarr) #[1 2]

#Find the symmetric difference of the set1 and set2: 

set1 = np.array([1, 2, 3, 4])
set2 = np.array([3, 4, 5, 6])

newarr = np.setxor1d(set1, set2, assume_unique=True)

print(newarr)
