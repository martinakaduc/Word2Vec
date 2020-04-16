import utils

# eval using gensim
print ('tài...')
utils.most_similar(positive=['ra'])
print ('chữ tài chữ mệnh...')
utils.most_similar(positive=['chữ', 'tài', 'mệnh'], negative=['tài'])
