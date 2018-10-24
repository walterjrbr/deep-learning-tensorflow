
import numpy as np 
import os 
from matplotlib import pyplot as plt 
import tensorflow as tf 

# criamos uma pasta para salvar o modelo
if not os.path.exists('tmp'): # se a pasta não existir
    os.makedirs('tmp') # cria a pasta

# baixa os dados na pasta criada e carrega os dados 
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("tmp/", one_hot=False)




graph = tf.Graph()
with graph.as_default():
    # criamos constante com valor 8
    b = tf.constant(8, name='b')
    print(b)

    # criamos constante com valor 0
    #b = tf.constant(0, name='b')
    #print(b)


    # definindo constantes
lr = 0.01 # taxa de aprendizado
n_iter = 2000 # número de iterações de treino
batch_size = 512 # qtd de imagens no mini-lote (para GDE)
n_inputs = 28 * 28 # número de variáveis (pixeis)
n_l1 = 1024 # número de neurônios da primeira camada
n_l2 = 1024 # número de neurônios da segunda camada
n_l3 = 1024 # número de neurônios da segunda camada

n_outputs = 10 # número classes (dígitos)

graph = tf.Graph() # cria um grafo
with graph.as_default(): # abre o grafo para que possamos colocar nós

    # Camadas de Inputs
    with tf.name_scope('input_layer'): # escopo de nome da camada de entrada
        x_input = tf.placeholder(tf.float32, [None, n_inputs], name='images')
        y_input = tf.placeholder(tf.int64, [None], name='labels')

    # Camada 1
    with tf.name_scope('first_layer'): # escopo de nome da primeira camada
        # variáveis da camada
        W1 = tf.Variable(tf.truncated_normal([n_inputs, n_l1]), name='Weights')
        b1 = tf.Variable(tf.zeros([n_l1]), name='bias')

        l1 = tf.add(tf.matmul(x_input, W1), b1, name='linear_transformation')
        l1 = tf.nn.relu(l1, name='relu')

    # Camada 2
    with tf.name_scope('second_layer'): # escopo de nome da segunda camada
        # variáveis da camada
        W2 = tf.Variable(tf.truncated_normal([n_l1, n_l2]), name='Weights')
        b2 = tf.Variable(tf.zeros([n_l2]), name='bias')

        l2 = tf.add(tf.matmul(l1, W2), b2, name='linear_transformation')
        l2 = tf.nn.relu(l2, name='relu')

        # Camada 3
    with tf.name_scope('third_layer'): # escopo de nome da segunda camada
        # variáveis da camada
        W3 = tf.Variable(tf.truncated_normal([n_l2, n_l3]), name='Weights')
        b3 = tf.Variable(tf.zeros([n_l3]), name='bias')

        l3 = tf.add(tf.matmul(l2, W3), b3, name='linear_transformation')
        l3 = tf.nn.relu(l3, name='relu')

    # Camada de saída
    with tf.name_scope('output_layer'): # escopo de nome da camada de saída
        # variáveis da camada
        Wo = tf.Variable(tf.truncated_normal([n_l3, n_outputs]), name='Weights')
        bo = tf.Variable(tf.zeros([n_outputs]), name='bias')

        scores = tf.add(tf.matmul(l3, Wo), bo, name='linear_transformation') # logits
        error = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=scores),
            name='error')

    # calcula acurácia
    correct = tf.nn.in_top_k(scores, y_input, 1) # calcula obs corretas (vetor bools V ou F)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # converte de bool para float32

    # otimizador
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(error)

    # inicializador
    init = tf.global_variables_initializer()

    # para salvar o modelo treinado
    saver = tf.train.Saver()

    # conferindo os nomes
    print(y_input)
    print(scores)
    print(W1)
    print(b1)
    print(W2)
    print(b2)
    print(W3)
    print(b3)




    # abrimos a sessão tf
with tf.Session(graph=graph) as sess:
    init.run() # iniciamos as variáveis

    # loop de treinamento
    for step in range(n_iter+1):

        # cria os mini-lotes
        x_batch, y_batch = data.train.next_batch(batch_size)

        # cria um feed_dict
        feed_dict = {x_input: x_batch, y_input: y_batch}

        # executa uma iteração de treino e calcula o erro
        l, _ = sess.run([error, optimizer], feed_dict=feed_dict)

        # mostra o progresso a cada 1000 iterações
        if step % 1000 == 0:

            x_valid, y_valid = data.validation.next_batch(512) # pega alguns dados de validação
            val_dict = {x_input: x_valid, y_input: y_valid} # monta o feed_dict

            # executa o nó para calcular a acurácia
            error_np, acc = sess.run([error, accuracy], feed_dict=val_dict)

            print('Erro de treino na iteração %d: %.2f' % (step, l))
            print('Erro de validação na iteração %d: %.2f' % (step, error_np))
            print('Acurácia de validação na iteração %d: %.2f\n' % (step, acc))

            # salva as variáveis do modelo
            saver.save(sess, "./tmp/deep_ann.ckpt")

def fully_conected_layer(inputs, n_neurons, name_scope, activations=None):
    '''Adiciona os nós de uma camada ao grafo TensorFlow'''
    
    n_inputs = int(inputs.get_shape()[1]) # pega o formato dos inputs
    with tf.name_scope(name_scope):
        
        # define as variáveis da camada
        with tf.name_scope('Parameters'):
            W = tf.Variable(tf.truncated_normal([n_inputs, n_neurons]), name='Weights')
            b = tf.Variable(tf.zeros([n_neurons]), name='biases')
            
            tf.summary.histogram('Weights', W) # para registrar o valor dos W
            tf.summary.histogram('biases', b) # para registrar o valor dos b
        
        # operação linar da camada
        layer = tf.add(tf.matmul(inputs, W), b, name='Linear_transformation')
        
        # aplica não linearidade, se for o caso
        if activations == 'relu':
            layer = tf.nn.relu(layer, name='ReLU')
        
        # para registar a ativação na camada
        tf.summary.histogram('activations', layer)
        
        return layer

logdir = 'logs' # nome pasta para salvar os arquivos de visualização

graph = tf.Graph()
with graph.as_default():

    # Camadas de Inputs
    with tf.name_scope('input_layer'):
        x_input = tf.placeholder(tf.float32, [None, n_inputs], name='images')

        y_input = tf.placeholder(tf.int64, [None], name='labels')

    # Camada 1
    l1 = fully_conected_layer(x_input, n_neurons=n_l1, name_scope='First_layer', activations='relu')

    # Camada 2
    l2 = fully_conected_layer(l1, n_neurons=n_l2, name_scope='Second_layer', activations='relu')

    # Camada 3
    l3 = fully_conected_layer(l2, n_neurons=n_l3, name_scope='Third_layer', activations='relu')

    # Camada de saída
    scores = fully_conected_layer(l3, n_neurons=n_outputs, name_scope='Output_layer') # logits

    # camada de erro
    with tf.name_scope('Error_layer'):
        error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=scores),
                               name='error')
        tf.summary.scalar('Cross_entropy', error) # para registrar a função custo

    with tf.name_scope("Accuracy"):
        correct = tf.nn.in_top_k(scores, y_input, 1) # calcula obs corretas
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # converta para float32
        tf.summary.scalar('Accuracy', accuracy) # para registrar a função custo

    # otimizador
    with tf.name_scope('Train_operation'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(error)

    # inicializador
    init = tf.global_variables_initializer()

    # para salvar o modelo treinado
    saver = tf.train.Saver()

    # para registrar na visualização
    summaries = tf.summary.merge_all() # funde todos os summaries em uma operação
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph()) # para escrever arquivos summaries

print("iniciando o grafo")



graph1 = tf.Graph()
with graph1.as_default():

    # Camadas de Inputs
    with tf.name_scope('input_layer'):
        x_input = tf.placeholder(tf.float32, [None, n_inputs], name='images')

        print(x_input)

        y_input = tf.placeholder(tf.int64, [None], name='labels')

    # Camada 1
    l1 = fully_conected_layer(x_input, n_neurons=n_l1, name_scope='First_layer', activations='relu')

    # Camada 2
    l2 = fully_conected_layer(l1, n_neurons=n_l2, name_scope='Second_layer', activations='relu')

    # Camada 3
    l3 = fully_conected_layer(l2, n_neurons=n_l3, name_scope='Third_layer', activations='relu')

    # Camada de saída
    scores = fully_conected_layer(l3, n_neurons=n_outputs, name_scope='Output_layer') # logits

    # camada de erro
    with tf.name_scope('Error_layer'):
        error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=scores),
                               name='error')
        tf.summary.scalar('Cross_entropy', error) # para registrar a função custo

    with tf.name_scope("Accuracy"):
        correct = tf.nn.in_top_k(scores, y_input, 1) # calcula obs corretas
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # converta para float32
        tf.summary.scalar('Accuracy', accuracy) # para registrar a função custo

    # otimizador
    with tf.name_scope('Train_operation'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(error)

    # inicializador
    init = tf.global_variables_initializer()
    
    # para salvar o modelo treinado
    saver = tf.train.Saver()
    
=======
>>>>>>> ff73c9e7ec4073032fbcc28663e9351cf924fc2a
