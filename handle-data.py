import pandas as pd

print('-------------------------------------------------------------------')
data = pd.read_csv('testdata.csv')
texts = data['Text']

lengths = []
words = []

for i in texts:
	splits = i.split()
	lengths.append(len(splits))
	for j in splits:
		words.append(j)

widx = list(set(words))

def delete_stop_chars(stop_chars, w):
	for s in stop_chars:
		for i in range(len(w)):
			if w[i].find(s) != -1:
				w[i] = w[i].replace(s, '')
	return w

def remove_elements(removable, w):
	for r in removable:
		reml = []
		for i in range(len(w)):
			if w[i].find(r) != -1:
				# w.pop(i)
				reml.append(i)
		reml.reverse()
		for k in reml:
			w.pop(k)
	for j in range(len(w)-1, 0, -1):
		if not w[j]:
			w.pop(j)
	return w

def append_important(w):
	i = [',', '.', ':', '!', '?', ';']
	for j in i:
		w.append(j)
	return w

stops = [',', '.', ':', '!', '?', ';', '/', '<', 'br', '>', '(', ')']
widx = delete_stop_chars(stops, widx)

rem = ['href']
widx = remove_elements(rem, widx)

widx = append_important(widx)
# widx = set(widx)
# print(widx)

def create_combined_text(txt):
	temp = [x for x in txt]
	t = ''
	return t.join(temp)

def split_text(txt):
	return txt.split()

def turn_into_references(txt, w):
	temp = []
	for i in txt:
		temp.append(w.index(i))
	return temp

# t = create_combined_text(texts)
# t = split_text(t)
t = delete_stop_chars(stops, words)
t = remove_elements(rem, t)
t = turn_into_references(t, widx)

print(t)