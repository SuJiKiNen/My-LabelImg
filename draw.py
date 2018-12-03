import matplotlib.pyplot as plt
import os
import numpy as np

f_loss = open('myvgg_losses','r').read()
f_accuracy = open('myvgg_accuracies','r').read()

loss_list = f_loss.split('\n')
accuracy_list = f_accuracy.split('\n')

losses = []
accuracies = []

for i in range(len(loss_list)):
    losses.append(float(loss_list[i]))
for i in range(len(accuracy_list)):
    accuracies.append(float(accuracy_list[i]))

x1 = range(1,len(loss_list)+1)
x2 = range(1,len(accuracy_list)+1)

plt.figure()
plt.plot(x1,losses)
plt.title('training without using pretrained weights')
min_loss_index = losses.index(min(losses))
x,y = min_loss_index,min(losses)
plt.text(x,y,'(%d,%.2f)'%(x,y),ha='center',va='bottom',fontsize=8)

plt.xlabel('steps')
plt.ylabel('losses')
plt.savefig('myvggloss.png')

plt.figure()
plt.plot(x2,accuracies)
plt.title('training without using pretrained weights')
plt.xlabel('validation')
plt.ylabel('accuracy')
plt.savefig('myvggaccuracy.png')

plt.show()

    