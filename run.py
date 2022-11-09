from distutils.log import set_verbosity
from operator import contains
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os
import yaml
from common.log import Logger
import time
import logging
import sys
from Model.Lenet import Model

best_acc = []
best_param = []
best_models = []    

class NeuroEvolution:

    def __init__(self, config):
        
        self.num_filters = 25
        self.num_epochs = 5
        self.batch_size = 1024

        self.create_dataset(config)

        self.num_elites = 3 
        self.out_dir = './Outdir'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        if config["parameter"].lower() not in ["fps","accuracy",'bandwidth']:
            print("The specified Parameter is not recognized.\n")
            #exit()
            raise SystemExit
        self.fitness_parameter=config["parameter"].lower()
        print("Fitness parameter has been set successfully.\n")
        

        self.device = 'cuda'
        self.device_test = 'cpu'

        self.exp_dir = os.path.join(config['checkpoint_dir'], config['exp_name']+'_'+str(int(time.time())))
    
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        
        os.mkdir(os.path.join(self.exp_dir, 'weights'))
        os.mkdir(os.path.join(self.exp_dir, 'onnx_weights'))
        

        self.logger = Logger(os.path.join(self.exp_dir, 'log.txt'), config)
        self.logger.set_names(['Generation\t', 'best_accuracy\t', 'best_throughput\t', 'fitness_scores\t'])

        self.result_config = os.path.join(self.exp_dir, 'result_config.yaml')

        self.generation = 0

    def create_dataset(self,config):
        
        if not os.path.exists(config['test']) or not os.path.exists(config['train']) or not os.path.exists(config['validation']):
            print("Specified path is incorrect or does not exist. Check the path specified for Train, Test and Validation\n")
            #exit()
            raise SystemExit
        if len(os.listdir(config['train']))>1 or len(os.listdir(config['test']))>1 or len(os.listdir(config['validation']))>1:
            print("The directory contains additional files other than the required data\n")
            #exit()
            raise SystemExit
        if os.listdir(config['train'])[0]==os.listdir(config['test'])[0]==os.listdir(config['validation'])[0]=='MNIST':
            self.train_dataset = mnist.MNIST(root=config['train'], train=True, transform=ToTensor())
            self.train_loader = DataLoader(self.train_dataset, batch_size=1024)
            self.test_dataset = mnist.MNIST(root=config['test'], train=False, transform=ToTensor())
            self.test_loader = DataLoader(self.test_dataset, batch_size=1024)
            self.validation_dataset= mnist.MNIST(root=config['validation'], train=False, transform=ToTensor())
            self.test_loader = DataLoader(self.test_dataset, batch_size=1024)
        elif os.listdir(config['train'])[0][-3:]==os.listdir(config['test'])[0][-3:]==os.listdir(config['validation'])[0][-3:]=="csv":
            pass
        else:
            print("All of the files are not in the same format. Change all the files in Train, Test and Validation to the same format\n")
            #exit()
            raise SystemExit
        print("Train, Test and Validation data have been loaded successfully\n")
        
        
    def create_population(self, population_size):

        population = []
        for num in range(population_size):
            population.append(Model(num_filters=self.num_filters))
        
        return population

    def train_population(self, population):

        

        count = 0
        for model in tqdm(population, 'Training..'):
            loss_fn = CrossEntropyLoss()
            sgd = SGD(model.parameters(), lr=1e-1)
            model = model.to(self.device)
            model.train()

            for current_epoch in range(self.num_epochs):    
                for idx, (train_x, train_label) in enumerate(self.train_loader):
                    
                    train_x = train_x.to(self.device)
                    train_label = train_label.to(self.device)

                    sgd.zero_grad()
                    predict_y = model(train_x.float())
                    loss = loss_fn(predict_y, train_label.long())
                    loss.backward()
                    sgd.step() 
                    
            count += 1

        return population

    def fitness(self, population):
        
        accuracies = []
        parameters = []
        throughputs = []

        count = 0
        for model in tqdm(population, 'Testing..'):

            all_correct_num = 0
            all_sample_num = 0
            model = model.to(self.device_test)
            model.eval()
            for idx, (test_x, test_label) in enumerate(self.test_loader):

                test_x = test_x.to(self.device_test)

                predict_y = model(test_x.float()).cpu().detach()

                
                predict_y = np.argmax(predict_y, axis=-1)
                current_correct_num = predict_y == test_label
                all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
                all_sample_num += current_correct_num.shape[0]
            
            input_names = ["MNIST"]
            output_names = [ "output" ]

            torch.onnx.export(model, test_x, os.path.join(self.exp_dir, "onnx_weights", "lenet.onnx"), verbose=False, 
                                input_names=input_names, output_names=output_names)
            output = os.system("benchmark_app -m "+os.path.join(self.exp_dir, "onnx_weights", "lenet.onnx") + " > output.txt")
            
            with open('output.txt', 'r') as f:
                content = f.readlines()

            throughput = content[-1]
            throughput = throughput.split(' ')
            throughput = float(throughput[1])
            throughputs.append(throughput)
            
            print(throughput)

            acc = all_correct_num / all_sample_num
            param = sum(p.numel() for p in model.parameters())

            accuracies.append(acc * 100)
            parameters.append(param)

            count += 1

        
        accuracies = np.array(accuracies) / 100
        throughputs = np.array(throughputs)

        fitness_scores = accuracies * throughputs

        para_d={"fps":throughputs,'accuracy':accuracies}
        
        elite_models = np.argsort(para_d[self.fitness_parameter])[::-1][:self.num_elites]
        
        print('\nBest Models : \n')
        for en,f in enumerate(elite_models):
            print(en, ' : ', accuracies[f], ' : ', throughputs[f],' : ', fitness_scores[f])
            self.logger.append([self.generation, accuracies[f], throughputs[f], fitness_scores[f]])
        global best_acc
        global best_param
        global best_models
        # Save models if those are better than previous best
        if len(best_models) == 0 and len(elite_models)>0:
            
            for f in elite_models:
                best_models.append(population[f])
                best_acc.append(accuracies[f])
                best_param.append(throughputs[f])
        
        else:
            
            i=j=0
            while(i<len(elite_models) and j<len(best_models)):
                if para_d[self.fitness_parameter][elite_models[i]]< best_param[j] if self.fitness_parameter=="fps" else best_acc[j]:
                    j+=1
                else:
                    if j==0:
                        best_models=np.concatenate([[population[elite_models[i]]],best_models[:-1]])
                        best_acc=np.concatenate([[accuracies[elite_models[i]]], best_acc[:-1]])
                        best_param=np.concatenate([[throughputs[elite_models[i]]], best_param[:-1]])
                    elif j==len(best_models)-1:
                        best_models=np.concatenate([best_models[:-1],[population[elite_models[i]]]])
                        best_acc=np.concatenate([ best_acc[:-1],[accuracies[elite_models[i]]]])
                        best_param=np.concatenate([ best_param[:-1],[throughputs[elite_models[i]]]])
                    else:
                        best_models=np.concatenate([best_models[:j],[population[elite_models[i]]],best_models[j:-1]])
                        best_acc=np.concatenate([best_acc[:j],[accuracies[elite_models[i]]],best_acc[j:-1]])
                        best_param=np.concatenate([best_param[:j],[throughputs[elite_models[i]]],best_param[j:-1]])
                    i+=1
                    self.logger.record_args('Best model updated...')
                    print('Best model updated...')
            print('\nOverall Best Models after updation: \n')
            for f in range(len(best_models)):
                print(en, ' : ', best_acc[f], ' : ', best_param[f],' : ', best_acc[f]*best_param[f])
                self.logger.append([self.generation, best_acc[f], best_param[f], best_acc[f]*best_param[f]])
                        

#             for f in elite_models:
#                 for i in range(len(self.best_models)):
#                     if accuracies[f]>=self.best_acc[i] and throughputs[f]<=self.best_param[i]:
#                         self.best_models[i] = population[f]
#                         self.best_acc[i] = accuracies[f]
#                         self.best_param[i] = throughputs[f]
                    
#                     self.logger.record_args('Best model updated...')
#                     print('Best model updated...')

        return elite_models


    def mutate(self, model):

        # Layer parameters
        num_filters = np.random.randint(1,self.num_filters)
        print("Mutation :")
        print(num_filters)
        new_model = Model(num_filters=num_filters)
        print("\n",new_model,"\n")
        return new_model


    def selection(self, population, best_models):
        new_generation = []
        for model_n in best_models:
            new_generation.append(model_n)
        
        for model_num in range(len(population)):
            print(model_num,population[model_num] in best_models,best_models)
            if population[model_num] in best_models:
                continue
            
            mutated_model = self.mutate(population[model_num])
            new_generation.append(mutated_model)

        return new_generation

    def save_best_models(self):
        global best_acc
        global best_param
        global best_models
        
        for i in range(len(best_models)):
            
            torch.save({'state_dict' : best_models[i].state_dict()},
						os.path.join(self.exp_dir, 'weights','best_%d_acc_%d_param_%d.pth.tar'%(i,best_acc[i]*100,best_param[i])))

    def save_config(self):
        global best_acc
        global best_param
        global best_models
        #dict_file = {}
        #for i in range(len(best_models)):
            
        #    dict_file['best_model_'+str(i)] = {}
        #    dict_file['best_model_'+str(i)]['num_filters'] = best_models[i].num_filters
        #    dict_file['best_model_'+str(i)]['accuracy'] = best_acc[i]
        #    dict_file['best_model_'+str(i)]['parameters'] = best_param[i]

        #with open(self.result_config, 'w+') as file:
        #    documents = yaml.dump(dict_file, file)
            
            
        
        file1 = open("result.txt","w")
        dict_file = {}
        for i in range(len(best_models)):
            file1.write('best_model_'+str(i)+"\n")
            file1.write(str(best_models[i])+"\n")
            file1.write('best_model_'+str(i)+' num_filters : '+str(best_models[i].num_filters)+"\n")
            file1.write('best_model_'+str(i)+' accuracy : '+str(best_acc[i])+"\n")
            file1.write('best_model_'+str(i)+' parameters : '+str(best_param[i])+"\n\n")
        file1.close()

def read_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def main():

    population_size = 3
    num_generations = 100

    fail=True
    try:
        with open('./config.yml') as f:
            config = yaml.safe_load(f)
        f.close()
        print("Config file loaded successfully\n")
        fail=False
    except FileNotFoundError:
        print("Config file does not exist\n")
    except ValueError:
        print("Data in the JSON file is not properly formatted\n")
    if fail:
        #exit()
        raise SystemExit
    if sorted(list(config.keys()))!=sorted(['model', 'test', 'train', 'validation', 'parameter','exp_name', 'checkpoint_dir']):
        if len(config.keys())>5:
            print("Additional keys provided other than Model, Test, Train, Validation and Parameter. Please remove all other keys.\n")
        else:
            print("The following key/keys is/are not present in the YML file : "+" , ".join(list(set(['model', 'test', 'train', 'validation', 'parameter','exp_name', 'checkpoint_dir'])-set(config.keys())))+"\n")
        #exit()
        raise SystemExit
    if config["model"].lower()=='lenet':
        from Model.Lenet import Model
    elif config["model"].lower()=='resnet':
        from Model.Resnet import Model
    else:
        print("Model Name Not Recognized\n")
        #exit()
        raise SystemExit
    print(config["model"].lower().title()," architecture loaded successfully\n",sep="")
    global best_models
    nevol = NeuroEvolution(config)
    population = nevol.create_population(population_size)
    for generation in range(num_generations):
        print('\nGeneration : ', generation)
        population = nevol.train_population(population)
        best_model = nevol.fitness(population)

        nevol.save_best_models()
        nevol.save_config()

        new_generation = nevol.selection(population, best_models)

        population = new_generation

        nevol.generation += 1


if __name__ == '__main__':
    main()
