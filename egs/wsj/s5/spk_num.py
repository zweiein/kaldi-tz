#!/usr/bin/env python

import sys, collections

#dir = 'data_fbank/train_10'
dir = sys.argv[1]

utt2spk_dict = collections.OrderedDict()
with open(dir + '/utt2spk', 'r') as utt2spks:
	for utt2spk in [line.strip().split(' ') for line in utt2spks]:
		utt2spk_dict[utt2spk[0]] = utt2spk[1]

utt2spk_num = open(dir + '/utt2spk_num', 'w')
id = 0
count = 0
max = 0
min = 10000
spk_pre = 'nospeaker'
for i in utt2spk_dict.keys():
		line = i
		if spk_pre != utt2spk_dict[i][:3]:
			id += 1
			spk_pre = utt2spk_dict[i][:3]
			if max < count:
				max = count
			if min > count and count > 0:
				min = count
			count = 0
		count += 1
		utt2spk_num.write(line + ' ' + str(id) + '\n')
utt2spk_num.close()

if max < count:
	max = count
if min > count and count > 0:
	min = count
print('max utts spoken by one person:')
print(max)
print('min utts spoken by one person:')
print(min)
