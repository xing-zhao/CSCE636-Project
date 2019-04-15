import tkinter as tk
from tkinter import *
import pickle
from tkinter import messagebox
from PIL import ImageTk, Image
import pandas as pd
import numpy as np
import warnings
import pickle
from time import time
from keras import backend as K
import argparse
import keras
import json
import glob, os
import webbrowser
import csv

warnings.filterwarnings('ignore')

class Xing_Rec:
	def __init__(self):
		self.dataset = 'ml-100k/ml-100k'
		#load data
		with open(self.dataset + '_Train_list.txt', "rb") as fp:
		    self.Train_list = pickle.load(fp)
		    
		with open(self.dataset + '_Vali_list.txt', "rb") as fp:
		    self.Vali_list = pickle.load(fp)
		    
		with open(self.dataset + '_Test_list.txt', "rb") as fp:
		    self.Test_list = pickle.load(fp)

		self.movie_index_to_id = json.load(open('ml-100k/Movie_index_to_id.json'))
		self.movie_id_to_index = {int(k):int(v) for v,k in self.movie_index_to_id.items()}

		self.all_user_index = []
		self.all_movie_index = []
		for elem in self.Train_list:
		    self.all_user_index.append(elem[0])
		    self.all_movie_index.append(elem[1])
		for elem in self.Vali_list:
		    self.all_user_index.append(elem[0])
		    self.all_movie_index.append(elem[1])
		for elem in self.Test_list:
		    self.all_user_index.append(elem[0])
		    self.all_movie_index.append(elem[1])
		    

		self.all_user_index = list(set(self.all_user_index))
		self.all_movie_index = list(set(self.all_movie_index))

		self.n_users = max(self.all_user_index)
		self.n_movies = max(self.all_movie_index)

		self.current_user_id = self.n_users


		self.like_length = 20

		self.movie_id_name = {}
		with open("ml-100k/movies.csv", "r") as infile:
			reader = csv.reader(infile)
			next(reader, None)  # skip the headers
			for row in reader:
				if int(row[0]) in self.all_movie_index:
					self.movie_id_name[int(row[0])] = row[1]


		self.movie_id_link = {}
		with open("ml-100k/links.csv", "r") as infile:
			reader = csv.reader(infile)
			next(reader, None)  # skip the headers
			for row in reader:
				if int(row[0]) in self.movie_id_name:
					self.movie_id_link[int(row[0])] = int(row[1])

		self.movie_id_with_poster = []
		os.chdir("ml20_posters/MLP-20M/")
		for file in glob.glob("*.jpg"):
			_id = int(file.split('.')[0])
			if _id in self.movie_id_name:
				self.movie_id_with_poster.append(_id)
		os.chdir("../../")

		self.random_like_movie = np.random.choice(self.movie_id_with_poster, self.like_length)
		self.random_dislike_movie = np.random.choice(self.movie_id_with_poster, self.like_length)


		self.window = tk.Tk()
		self.window.title('Welcome to Movie Recommandation - Xing Zhao')
		self.window.geometry('450x300')

		# welcome iamge
		self.canvas = tk.Canvas(self.window, height = 200, width = 500)
		self.image_file = tk.PhotoImage(file= 'welcome.gif')
		self.iamge = self.canvas.create_image(0,0, anchor = 'nw', image = self.image_file)
		self.canvas.pack(side = 'top')

		# user information
		tk.Label(self.window, text = 'User name: ').place(x = 50, y = 150)
		tk.Label(self.window, text = 'Password: ').place(x = 50, y = 190)

		self.var_usr_name = tk.StringVar()
		self.var_usr_name.set('exam@python.com')

		self.entry_usr_name = tk.Entry(self.window, textvariable = self.var_usr_name)
		self.var_usr_pwd = tk.StringVar()
		self.entry_usr_name.place (x = 160, y = 150)

		self.entry_usr_pwd = tk.Entry(self.window, 
			textvariable = self.var_usr_pwd,
			show = '*')
		self.entry_usr_pwd.place (x = 160, y = 190)


		# login and sign up button
		self.btn_login = tk.Button(self.window, text = 'Login', command = self.usr_login)
		self.btn_login.place(x = 170, y = 230)

		self.btn_singup = tk.Button(self.window, text = 'Sign up', command = self.usr_signup)
		self.btn_singup.place(x = 270, y = 230)

		self.all_like_movie = self.index_to_poster(self.random_like_movie)
		self.all_dislike_movie = self.index_to_poster(self.random_dislike_movie)

		self.window.mainloop()

	def index_to_poster(self, index):
		all_m = []
		for i in index:
			_m = Image.open("ml20_posters/MLP-20M/" + str(i) + ".jpg")
			_m = _m.resize((100,150), Image.ANTIALIAS)
			_m = ImageTk.PhotoImage(_m)
			all_m.append(_m)
		return all_m

	def usr_login(self):
		usr_name = self.var_usr_name.get()
		usr_pwd = self.var_usr_pwd.get()

		try:
			with open('usrs_info.pickle', 'rb') as usr_file:
				usrs_info = pickle.load(usr_file)
		except IOError:
			with open('usrs_info.pickle', 'wb') as usr_file:
				usrs_info = {'admin': 'admin'}
				pickle.dump(usrs_info, usr_file)

		if usr_name in usrs_info:
			if usr_pwd == usrs_info[usr_name]:
				messagebox.showinfo(title = 'Welcome!',
				message = 'Welcome Back, ' + usr_name + '!')

				self.Recommandation()


			else:
				messagebox.showerror(message = 'Error! Your password is wrong, try again!')

		else:
			is_sign_up = messagebox.askyesno('Welcome', 'You have not sign up yet. Sign up today?')
			if is_sign_up:
				usr_signup()

	def usr_signup(self):
		def sign_to_Recommandation():
			newp = new_pwd.get()
			newpf = new_pwd_confirm.get()
			newn = new_name.get()

			with open('usrs_info.pickle', 'rb') as usr_file:
				exist_usr_info = pickle.load(usr_file)

			if newp != newpf:
				messagebox.showerror('Error', 'Password and confirm password must be the same!')
			elif newn in exist_usr_info:
				messagebox.showerror('Error', 'The user has already signed up!')
			else:
				exist_usr_info[newn] = newp
				with open('usrs_info.pickle', 'wb') as usr_file:
					pickle.dump(exist_usr_info, usr_file)
				messagebox.showinfo('Welcome', 'You have successfully signed up!')
				window_signup.destroy()

		window_signup = tk.Toplevel(self.window)
		window_signup.title('Sign up window')
		window_signup.geometry('350x200')

		new_name = tk.StringVar()
		new_name.set('exam@python.com')
		tk.Label(window_signup, text = 'User name').place(x = 10, y = 10)
		entry_new_name = tk.Entry(window_signup, textvariable = new_name)
		entry_new_name.place(x = 150, y = 10)

		new_pwd = tk.StringVar()
		tk.Label(window_signup, text = 'Password').place(x = 10, y = 50)
		entry_new_pwd = tk.Entry(window_signup, 
			textvariable = new_pwd,
			show = '*')
		entry_new_pwd.place(x = 150, y = 50)

		new_pwd_confirm = tk.StringVar()
		tk.Label(window_signup, text = 'Confirm Password').place(x = 10, y = 90)
		entry_new_pwd_confirm = tk.Entry(window_signup, 
			textvariable = new_pwd_confirm,
			show = '*')
		entry_new_pwd_confirm.place(x = 150, y = 90)

		btn_confirm_signup = tk.Button(window_signup,
			text = 'Sign up',
			command = sign_to_Recommandation)
		btn_confirm_signup.place(x = 150, y = 130)

	def selected_checkbox(self):
		pass

	def Recommandation(self):
		
		window_rec = tk.Toplevel(self.window)
		window_rec.title('Create Your profile')
		window_rec.geometry('1700x1100')


		tk.Label(window_rec, text = 'Please Select Movies You Love: ').place(x = 25, y = 25)

		self.all_like_var = []
		for i in range(self.like_length):
			self.all_like_var.append(tk.IntVar())

		self.all_dislike_var = []
		for i in range(self.like_length):
			self.all_dislike_var.append(tk.IntVar())

		for i in range(self.like_length):

			_movie1_lable = tk.Label(window_rec, image = self.all_like_movie[i])
			_movie1_lable.place(x =100 + (i%10) * 150, y = 75 + int(i / 10) * 180)
			
			_text = self.movie_id_name[self.random_like_movie[i]][:13]
			c1 = tk.Checkbutton(window_rec, text=_text, variable=self.all_like_var[i], onvalue=1, offvalue=0,
		                    command=self.selected_checkbox)
			c1.place(x = 100+ (i%10) * 150, y = 225+ int(i / 10) * 180)



		tk.Label(window_rec, text = 'Please Select Movies You Donot Like: ').place(x = 25, y = 450)

		for i in range(self.like_length):

			_movie1_lable = tk.Label(window_rec, image = self.all_dislike_movie[i])
			_movie1_lable.place(x =100 + (i%10) * 150, y = 500 + int(i / 10) * 180)
			var1 = tk.IntVar()
			_text = self.movie_id_name[self.random_dislike_movie[i]][:13]
			c1 = tk.Checkbutton(window_rec, text=_text, variable=self.all_dislike_var[i], onvalue=1, offvalue=0,
		                    command=self.selected_checkbox)
			c1.place(x = 100+ (i%10) * 150, y = 650+ int(i / 10) * 180)


		btn_rec = tk.Button(window_rec, text = 'Submit', command = self.submit_result)
		btn_rec.place(x = 550, y = 900)

		btn_exit = tk.Button(window_rec, text = 'Exit', command = exit)
		btn_exit.place(x = 1000, y = 900)

	def submit_result(self):
		self.user_like = []
		for i in range(len(self.all_like_var)):
			if (self.all_like_var[i].get() == 1):
				self.user_like.append(self.random_like_movie[i])

		self.user_dislike = []
		for i in range(len(self.all_dislike_var)):
			if (self.all_dislike_var[i].get() == 1):
				self.user_dislike.append(self.random_dislike_movie[i])


		self.current_user_preference_index = []
		self.current_user_already_seen_index = []
		for elem in self.user_like:
			self.current_user_preference_index.append((self.current_user_id, self.movie_id_to_index[elem], 5, 1))
			self.current_user_already_seen_index.append(self.movie_id_to_index[elem])

		for elem in self.user_dislike:
			self.current_user_preference_index.append((self.current_user_id, self.movie_id_to_index[elem], 1, 0))
			self.current_user_already_seen_index.append(self.movie_id_to_index[elem])


		self.final_recommand = self.predicting()
		self.result()


	def predicting(self):
		num_epochs = 60
		warnings.filterwarnings('ignore')

		n_latent_factors_user = 30
		n_latent_factors_movie = 30
		n_latent_factors_mf = 10


		movie_input = keras.layers.Input(shape=[1],name='Item-OneHot')
		movie_embedding_mlp = keras.layers.Embedding(self.n_movies + 1, n_latent_factors_movie, name='MLP-Item-Vector')(movie_input)
		movie_vec_mlp = keras.layers.Flatten(name='MLP-Item-Flatten')(movie_embedding_mlp)
		movie_vec_mlp = keras.layers.Dropout(0.5, name='MLP-Item-Flatten-Dropout')(movie_vec_mlp)

		movie_embedding_mf = keras.layers.Embedding(self.n_movies + 1, n_latent_factors_mf, name='MF-Item-Vector')(movie_input)
		movie_vec_mf = keras.layers.Flatten(name='MF-Item-Flatten')(movie_embedding_mf)
		movie_vec_mf = keras.layers.Dropout(0.5, name='MF-Item-Flatten-Dropout')(movie_vec_mf)


		user_input = keras.layers.Input(shape=[1],name='User-OneHot')
		user_vec_mlp = keras.layers.Flatten(name='MLP-User-Flatten')(keras.layers.Embedding(self.n_users + 2, n_latent_factors_user,name='MLP-User-Vector')(user_input))
		user_vec_mlp = keras.layers.Dropout(0.5, name='MLP-User-Flatten-Dropout')(user_vec_mlp)

		user_vec_mf = keras.layers.Flatten(name='MF-User-Flatten')(keras.layers.Embedding(self.n_users + 2, n_latent_factors_mf,name='MF-User-Vector')(user_input))
		user_vec_mf = keras.layers.Dropout(0.5, name='MF-User-Flatten-Dropout')(user_vec_mf)


		concat = keras.layers.merge([movie_vec_mlp, user_vec_mlp], mode='concat',name='MLP-Concat')
		concat_dropout = keras.layers.Dropout(0.5)(concat)
		dense = keras.layers.Dense(100,name='Fully-Connected')(concat_dropout)
		dense_batch = keras.layers.BatchNormalization(name='Batch')(dense)
		dropout_1 = keras.layers.Dropout(0.5,name='Dropout-1')(dense_batch)
		dense_2 = keras.layers.Dense(100,name='Fully-Connected-1')(dropout_1)
		dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)


		dropout_2 = keras.layers.Dropout(0.5,name='Dropout-2')(dense_batch_2)
		dense_3 = keras.layers.Dense(100,name='Fully-Connected-2')(dropout_2)
		dense_4 = keras.layers.Dense(50,name='Fully-Connected-3', activation='relu')(dense_3)

		pred_mf = keras.layers.merge([movie_vec_mf, user_vec_mf], mode='mul',name='Dot')


		pred_mlp = keras.layers.Dense(30, activation='relu',name='Activation')(dense_4)

		combine_mlp_mf = keras.layers.merge([pred_mf, pred_mlp], mode='concat',name='Concat-MF-MLP')
		result_combine = keras.layers.Dense(100,name='Combine-MF-MLP')(combine_mlp_mf)
		deep_combine = keras.layers.Dense(100,name='Fully-Connected-4')(result_combine)


		result = keras.layers.Dense(1,name='Predicted-Rating')(deep_combine)


		model = keras.Model([user_input, movie_input], result)
		opt = keras.optimizers.Adam(lr =0.05)
		model.compile(optimizer=opt,loss= 'mean_squared_error',metrics=['mae'])


		Train_list_user = []
		Train_list_movie = []
		Train_list_rating = []
		for elem in self.Train_list:
		    Train_list_user.append(elem[0])
		    Train_list_movie.append(elem[1])
		    Train_list_rating.append(elem[2])

		for elem in self.current_user_preference_index:
			Train_list_user.append(elem[0])
			Train_list_movie.append(elem[1])
			Train_list_rating.append(elem[2])

		Train_list_user = np.array(Train_list_user) 
		Train_list_movie = np.array(Train_list_movie)   
		Train_list_rating = np.array(Train_list_rating)   

		    
		Vali_list_user = []
		Vali_list_movie = []
		Vali_list_rating = []
		for elem in self.Vali_list:
		    Vali_list_user.append(elem[0])
		    Vali_list_movie.append(elem[1])
		    Vali_list_rating.append(elem[2])
		Vali_list_user = np.array(Vali_list_user) 
		Vali_list_movie = np.array(Vali_list_movie)   
		Vali_list_rating = np.array(Vali_list_rating) 
		    
		Test_list_user = []
		Test_list_movie = []
		Test_list_rating = []
		for elem in self.Test_list:
		    Test_list_user.append(elem[0])
		    Test_list_movie.append(elem[1])
		    Test_list_rating.append(elem[2])
		Test_list_user = np.array(Test_list_user) 
		Test_list_movie = np.array(Test_list_movie)   
		Test_list_rating = np.array(Test_list_rating) 


		best_loss = float("inf")
		best_mae = float("inf")

		train_loss_values = []
		vali_loss_values = []
		train_mae_values = []
		vali_mae_values = []

		for epoch in range(num_epochs):
		    print ('Iteration ' + str(epoch))
		    t1 = time()
		    history = model.fit([Train_list_user, Train_list_movie], Train_list_rating, 
		                        verbose=0, 
		                        batch_size = 16384)
		    train_loss = history.history['loss'][0]
		    train_loss_values.append(train_loss)
		    train_mae = history.history['mean_absolute_error'][0]
		    train_mae_values.append(train_mae)
		    #print train_loss, train_mae
		    t2 = time()
		    print ('\tTraining [%.1f s]: MSE = %.4f, MAE = %.4f' % (t2-t1, train_loss, train_mae))
		    
		    loss, mae = model.evaluate([Vali_list_user, Vali_list_movie], Vali_list_rating)
		    vali_loss_values.append(loss)
		    vali_mae_values.append(mae)
		    
		    print('\tValidation [%.1f s]: MSE = %.4f, MAE = %.4f ' % (time()-t2, loss, mae))
		    if loss < best_loss:
		        best_mae, best_loss, best_iter = mae, loss, epoch
		        best_model_name = '%s_eNCF_%d.h5' %(self.dataset,time())
		        model.save_weights(best_model_name, overwrite=True)
		        

		print("End. Best Iteration %d:  MSE = %.4f, MAE = %.4f. " %(best_iter, best_loss, best_mae))


		print ('\nTesting on Best Model:')
		model.load_weights(best_model_name)


		#remove all h5 files
		os.chdir("ml-100k")
		for file in glob.glob("*.h5"):
			os.remove(file)
		os.chdir("../")


		test_mse, test_mae = model.evaluate([Test_list_user, Test_list_movie], Test_list_rating)

		print ("Testing Result: MSE: %.4f, MAE: %.4f" % (test_mse, test_mae))


		movie_id_with_poster_index = []

		for elem in self.movie_id_with_poster:
			if elem in self.movie_id_to_index:
				movie_id_with_poster_index.append(self.movie_id_to_index[elem])

		
		mm = np.array([elem for elem in np.array(movie_id_with_poster_index) if elem not in self.current_user_already_seen_index])
		uu=	 np.ones(mm.shape[0]) * self.current_user_id
		predict_uu = model.predict([uu, mm]).reshape((1,mm.shape[0]))

		top = [movie_id_with_poster_index[_iii] for _iii in predict_uu[0].argsort()[-10:][::-1]]
		top_id = [self.movie_index_to_id[str(i)] for i in top]

		return top_id


	def result(self):
		window_result = tk.Toplevel(self.window)
		window_result.title('Recommandation')
		window_result.geometry('900x550')

		tk.Label(window_result, text = 'Recommanding Movies for you').place(x = 25, y = 25)
		

		self.all_recommand_movie = self.index_to_poster(self.final_recommand)

		def callback1(btn):
			tid = str(btn.cget('textvariable'))
			_zero = '0' * (7-len(tid))
			
			link = 'http://www.imdb.com/title/tt' + _zero + tid
			print (link)

			webbrowser.open_new(link)

		for i in range(10):
			_movie1_lable = tk.Label(window_result, image = self.all_recommand_movie[i])
			_movie1_lable.place(x =100 + (i%5) * 150, y = 75 + int(i / 5) * 200)
			_link_id = str(self.movie_id_link[self.final_recommand[i]])
			_text = self.movie_id_name[self.final_recommand[i]][:13] 
			_name = tk.Button(window_result, text = _text, width=11, textvariable = _link_id)
			_name.place(x = 100+ (i%5) * 150, y = 235+ int(i / 5) * 200)
			_name.configure(command = lambda btn = _name: callback1(btn))

		btn_exit = tk.Button(window_result, text = 'Exit', command = exit)
		btn_exit.place(x = 440, y = 480)



if __name__ == "__main__":
	gui = Xing_Rec()

