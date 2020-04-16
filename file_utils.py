import re

# load doc into memory
def load_data(filename):
	# open the file as read only
	file = open(filename, 'r', encoding='utf8')
	# read all text
	text = file.read()
	# close the file
	file.close()

	return text

# save tokens to file, one dialog per line
def save_data(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w', encoding='utf8')
	file.write(data)
	file.close()
