# AI_Number-Recognition-Neural-Network
Uses http://neuralnetworksanddeeplearning.com/chap1.html as a base.\
Number recognition neural network used for Artifiical Intelligence course.

Necessary Python modules:
numpy
imageio

To run:

python main.py -h	# For list of commands

Example:

python main.py -a 3.0 -e 30 -s -b 10

IMPORTANT NOTE: Running with -r and -s flag will initialize\
random weights and save them to file losing any previously\
saved weights

python main.py -u	# Uses image.png as input, users can\
			# draw whatever number in image.png\
			# which will then be used in the network\
			# and output a result (not always correct)

Also not possible to insert own test_data at the moment.

Defaults:\
user = false\
alpha = 3.0\
epochs = 30\
random = false\
test = false\
save = false\
batch = 10
