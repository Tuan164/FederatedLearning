def partition_data(X, y, client_ids, num_iterations, len_per_iteration):
    
    client_datasets = {client_id: None for client_id in client_ids}

    last_index = 0  # where to start the next client's dataset from
    # partition each client its data
    for client_id in client_ids:
        datasets_i = {}  # datasets for client i
        start_idx = last_index
        last_index += num_iterations * len_per_iteration  # where this client's datasets will end
        for j in range(1, num_iterations+1):
            end_indx = start_idx + len_per_iteration * j
                
            #print('From {} to {}'.format(start_idx, end_indx))
            X_ij = X[start_idx:end_indx]
            y_ij = y[start_idx:end_indx]

            datasets_i[j] = (X_ij, y_ij)

        client_datasets[client_id] = datasets_i
    return client_datasets

