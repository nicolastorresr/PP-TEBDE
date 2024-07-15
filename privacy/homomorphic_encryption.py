import numpy as np
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import torch

class HomomorphicEncryption:
    def __init__(self, config):
        self.HE = Pyfhel()
        self.HE.contextGen(scheme='bfv', n=2**14, t_bits=20)
        self.HE.keyGen()
        self.config = config

    def encrypt_tensor(self, tensor):
        """
        Encrypt a PyTorch tensor using homomorphic encryption.
        """
        # Convert tensor to numpy array
        array = tensor.numpy()
        
        # Flatten the array
        flat_array = array.flatten()
        
        # Scale and round the values to integers
        scaled_array = np.round(flat_array * self.config.scaling_factor).astype(np.int64)
        
        # Encrypt the scaled array
        encrypted_array = [self.HE.encrypt(PyPtxt(x, self.HE)) for x in scaled_array]
        
        return encrypted_array, array.shape

    def decrypt_tensor(self, encrypted_array, original_shape):
        """
        Decrypt an encrypted array back to a PyTorch tensor.
        """
        # Decrypt the array
        decrypted_flat = np.array([self.HE.decrypt(x) for x in encrypted_array])
        
        # Rescale the values
        rescaled_flat = decrypted_flat / self.config.scaling_factor
        
        # Reshape to original shape
        decrypted_array = rescaled_flat.reshape(original_shape)
        
        # Convert back to PyTorch tensor
        return torch.from_numpy(decrypted_array).float()

    def add_encrypted(self, encrypted_a, encrypted_b):
        """
        Add two encrypted arrays element-wise.
        """
        return [self.HE.add(a, b) for a, b in zip(encrypted_a, encrypted_b)]

    def multiply_plain(self, encrypted_a, plain_b):
        """
        Multiply an encrypted array by a plain scalar.
        """
        scaled_b = int(round(plain_b * self.config.scaling_factor))
        return [self.HE.multiply_plain(a, scaled_b) for a in encrypted_a]

    def mean_encrypted(self, encrypted_tensors):
        """
        Compute the mean of multiple encrypted tensors.
        """
        sum_encrypted = encrypted_tensors[0]
        for tensor in encrypted_tensors[1:]:
            sum_encrypted = self.add_encrypted(sum_encrypted, tensor)
        
        n = len(encrypted_tensors)
        mean_encrypted = self.multiply_plain(sum_encrypted, 1/n)
        
        return mean_encrypted

    def secure_aggregation(self, model_updates):
        """
        Perform secure aggregation of model updates using homomorphic encryption.
        """
        encrypted_updates = []
        shapes = []
        
        # Encrypt all model updates
        for update in model_updates:
            encrypted_update = {}
            for key, tensor in update.items():
                encrypted_tensor, shape = self.encrypt_tensor(tensor)
                encrypted_update[key] = encrypted_tensor
                shapes.append((key, shape))
            encrypted_updates.append(encrypted_update)
        
        # Aggregate encrypted updates
        aggregated_update = {}
        for key in encrypted_updates[0].keys():
            tensors_to_aggregate = [update[key] for update in encrypted_updates]
            aggregated_update[key] = self.mean_encrypted(tensors_to_aggregate)
        
        # Decrypt aggregated update
        decrypted_update = {}
        for (key, shape), encrypted_tensor in zip(shapes, aggregated_update.values()):
            decrypted_update[key] = self.decrypt_tensor(encrypted_tensor, shape)
        
        return decrypted_update
