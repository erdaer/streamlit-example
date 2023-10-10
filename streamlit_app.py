import streamlit as st

import numpy as np

import matplotlib.pyplot as plt

 

def compute_q_ensemble(K, H, r, psi):

    # Compute q for each set of parameters

    q_ensemble = (2 * np.pi * K * H) / (np.log(2 * H / r) + psi)

    return q_ensemble

 

def compute_q_inj_ensemble(K, H, r, psi, K_inj_factor, tau):

    # Compute K_inj as a ratio between K and K_inj_factor

    K_inj = K / K_inj_factor

 

    # Compute q_inj for each set of parameters

    q_inj_ensemble = (2 * np.pi * K * H) / (np.log(2 * H / r) + ((K / K_inj - 1) * np.log(1 + tau / r)) + psi)

    return q_inj_ensemble

 

# Use Streamlit widgets to get user input

n_samples = st.sidebar.slider('Number of samples', 1000, 10000, 5000)

 

# Define the distribution type for each parameter

distribution_type = {

    'K': st.sidebar.selectbox('Distribution type for K', ['uniform', 'normal']),

    'H': st.sidebar.selectbox('Distribution type for H', ['uniform', 'normal']),

    'r': st.sidebar.selectbox('Distribution type for r', ['uniform', 'normal']),

    'psi': st.sidebar.selectbox('Distribution type for psi', ['uniform', 'normal']),

    'K_inj_factor': st.sidebar.selectbox('Distribution type for K_inj_factor', ['uniform', 'normal']),

    'tau': st.sidebar.selectbox('Distribution type for tau', ['uniform', 'normal'])

}

st.image("inflowTunnels.png")

# Define the bounds or mean and std dev for each parameter
col1, col2, col3 = st.columns(3)

param_values = {}

for param in distribution_type.keys():

    if distribution_type[param] == 'uniform':

        #mm_values = st.slider(f'Enter min value for {param}',1,100,(2,50))
        #param_values[param] = (float(mm_value[0]), float(mm_value[1]))
     
        max_value = col2.text_input(f'Enter max value for {param}',2)
        min_value = col1.text_input(f'Enter min value for {param}',1)
        param_values[param] = (float(min_value), float(max_value))

       
    elif distribution_type[param] == 'normal':

        mean_value = col1.text_input(f'Enter mean value for {param}',1)

        stddev_value = col2.text_input(f'Enter std dev value for {param}',1)

        param_values[param] = (float(mean_value), float(stddev_value))
        
        with col3:
             st.write('test')#plt.plot([0,1],[0,1],'-o')

# Generate random samples for K, H, r, psi, K_inj_factor and tau

parameters = ['K', 'H', 'r', 'psi', 'K_inj_factor', 'tau']

samples = {}

for param in parameters: 
    if distribution_type[param] == 'uniform':
            if param == 'K':     
                st.write('K as log')
                samples[param] = np.exp(np.random.uniform(low=np.log(param_values[param][0]), high=np.log(param_values[param][1]), size=n_samples))
            else:
                samples[param] = np.random.uniform(low=param_values[param][0], high=param_values[param][1], size=n_samples)             

    elif distribution_type[param] == 'normal':

        samples[param] = np.random.normal(loc=param_values[param][0], scale=param_values[param][1], size=n_samples)

 

# Call the function with the generated samples

q_ensemble = compute_q_ensemble(samples['K'], samples['H'], samples['r'], samples['psi'])

q_inj_ensemble = compute_q_inj_ensemble(samples['K'], samples['H'], samples['r'], samples['psi'], samples['K_inj_factor'], samples['tau'])

 

# Create a new figure

plt.figure()

 

# Plot histogram for q_ensemble

plt.hist(q_ensemble, bins=50, alpha=0.5, label='q')

 

# Plot histogram for q_inj_ensemble

plt.hist(q_inj_ensemble, bins=50, alpha=0.5, label='q_inj')

 

# Add title and labels

plt.title('Histograms for q and q_inj')

plt.xlabel('Value')

plt.ylabel('Frequency')

 

# Add legend

plt.legend(loc='upper right')

 

# Display the plot in Streamlit

st.pyplot(plt)
