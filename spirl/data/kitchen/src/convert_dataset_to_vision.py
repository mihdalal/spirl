
import h5py
import numpy as np
import os
import cv2
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm
# dataset_vision = h5py.File('data/%s.hdf5' % 'kitchen-mixed-v0-vision','r+')

# env = gym.make("kitchen-mixed-v0")
# dataset = env.get_dataset()
# arr = []
# env.reset()
# ctr = 0
# for i, (action, done) in enumerate(tqdm(zip(dataset['actions'], dataset['terminals']), total=dataset['actions'].shape[0])):
#     if done:
#         env.reset()
#     o, r, d, info = env.step(action)
#     im = env.render('rgb_array').astype(np.uint8)
#     arr.append(im)

# image_data = np.array(arr).astype(np.uint8)
# dataset['images'] = image_data
# print(dataset.keys(), print(dataset['images'].shape))
# save_filename = 'data/kitchen-vision/%s.hdf5' % 'kitchen-mixed-v0-vision-64'
# print('Saving dataset to %s.' % save_filename)
# h5_dataset = h5py.File(save_filename, 'w')
# for key in dataset:
#     h5_dataset.create_dataset(key, data=dataset[key], compression='gzip')
# print('Done.')

def append_dataset(dataset_name, images, observations, rewards, terminals, actions, infos):
    import d4rl
    import gym
    env = gym.make(dataset_name)
    env.reset()
    dataset = env.get_dataset()
    for o, r, t, a, i in tqdm(zip(dataset['observations'], dataset['rewards'], dataset['terminals'], dataset['actions'], dataset['infos']), total=dataset['actions'].shape[0]):
        if t:
            env.reset()
        env.step(a)
        im = env.render('rgb_array').astype(np.uint8)
        images.append(im)
        observations.append(o)
        rewards.append(r)
        terminals.append(t)
        actions.append(a)
        infos.append(i)
    return images, observations, rewards, terminals, actions, infos
images, observations, rewards, terminals, actions, infos = [], [], [], [], [], []
images, observations, rewards, terminals, actions, infos = append_dataset('kitchen-complete-v0', images, observations, rewards, terminals, actions, infos)
images, observations, rewards, terminals, actions, infos = append_dataset('kitchen-partial-v0', images, observations, rewards, terminals, actions, infos)
images, observations, rewards, terminals, actions, infos = append_dataset('kitchen-mixed-v0', images, observations, rewards, terminals, actions, infos)

dataset = dict(
    images=np.array(images), observations=np.array(observations), rewards=np.array(rewards), terminals=np.array(terminals), actions=np.array(actions), infos=infos,
)
print(dataset.keys(), print(dataset['images'].shape))
save_filename = 'data/kitchen-vision/%s.hdf5' % 'kitchen-total-v0-vision-64'
print('Saving dataset to %s.' % save_filename)
h5_dataset = h5py.File(save_filename, 'w')
for key in dataset:
    h5_dataset.create_dataset(key, data=dataset[key], compression='gzip')
print('Done.')
