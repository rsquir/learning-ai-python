from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']

label_encoder.fit(input_classes)
print("Class mapping:\n")
for i, item in enumerate(label_encoder.classes_):
	print(item, '-->', i, "\n")

labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print("Labels =\n", labels)
print("Encoded labels =\n", list(encoded_labels))

encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("Encoded labels=\n", encoded_labels)
print("Decoded labels=\n", list(decoded_labels))