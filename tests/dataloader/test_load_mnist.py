from dataloader.load_mnist import load_mnist

def test_load_mnist():
    try:
        ds_train, ds_test = load_mnist()
    except:
        raise
    
    return

def test_dataset_not_empty():
    ds_train, ds_test = load_mnist()
    assert ds_train is not None
    assert ds_test is not None
    return

def test_mnist_ds_has_correct_shape():
    ds_train, ds_test = load_mnist()
    for (context_x, context_y, target_x), target_y in ds_train.take(1):
        assert context_x.shape == (32, 10, 2)
        assert context_y.shape == (32, 10, 1)
        assert target_x.shape == (32, 784, 2)
        assert target_y.shape == (32, 784, 1)
    return
