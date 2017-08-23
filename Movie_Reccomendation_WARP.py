import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#importing data from movielense

data = fetch_movielens(min_rating=5.0)

print(repr(data['train']))
print(repr(data['test']))

#build a model to fit the data in

model = LightFM(loss ='warp')
model.fit(data['test'],epochs = 10, num_threads = 2)


#function to create recommendations
def recommendation(model,data,user_ids):
	no_ids,no_items = data['train'].shape
	
	for user_id in user_ids:

		#finding movies rated 5.0 and above by the users
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
			
		#fitting in the model
		score = model.predict(user_id,np.arange(no_items))

		top_items = data['item_labels'][np.argsort(-score)]
		
		print("User %s" % user_id)
		print("     Known positives:")

		for x in known_positives[:3]:
			print("        %s" % x)
		
		print("     Recommended:")

		for x in top_items[:3]:
			print("        %s" % x)		
		
recommendation(model, data, [3, 25, 450])
		


