import pandas as pd
import os
csv_file = 'collision_labels.csv'
root_dir = 'collision/'
root_data = 'data/'

# obrazki = os.listdir('labeled_data')
# for s in obrazki:
#     os.rename(os.path.join('labeled_data/', s), os.path.join('labeled_data/', s+'.jpg'))
# print(obrazki)


dataset = pd.read_csv(os.path.join(root_dir, csv_file), index_col=0)
#
# for i in range(len(dataset)):
#     img_name = dataset.iloc[i, 0]
#     img_name += '.jpg'
#     dataset.set_value(i, 'img_name', img_name)
#
# print(dataset)
# dataset.to_csv('newcsv.csv')


n=0
rows_to_drop = []
for i in range(len(dataset)):
    img_name  = dataset.iloc[i, 0]
    img_path = os.path.join(root_dir, img_name)
    if not os.path.isfile(img_path):
        n += 1
        rows_to_drop.append(i)
print(n)
dataset = dataset.drop(rows_to_drop)
print(len(dataset))
dataset.to_csv(os.path.join(root_dir, 'collision_labels_clean.csv'))