# items = ['4 Beyond Classical Search',
#          'Artificial Intelligence A Modern Approach',
#          'Artificial Intelligence with Python',
#          'Data Analysis with Python Course',
#          'Deep Learning with Python',
#          'Deep Learning with PyTorch',
#          'Hands on Machine Learning with Scikitlearn Keras and TensorFlow',
#          'Machine Learning with Python',
#          'Neural Networks from Scratch in Python',
#          'Pattern Recognition And Machine Learning',
#          'Scikitlearn Crash Course']

items = ['13 Quantifying Uncertainty']
new_items = []

for item in items:
    new_items.append(item.replace(' ', '_'))


print(new_items)
# create file name extension
