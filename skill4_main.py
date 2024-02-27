import preprocessing as ic
import skill4_ann as dp

if __name__ == '__main__':
    images_folder_path = 'train'
    imgdg = ic.preprocess_data()
    imgdg.visualize_images(images_folder_path, nimages=2)
    image_df, train, label = imgdg.preprocess(images_folder_path)
    image_df.to_csv("image_df.csv")
    tr_gen, tt_gen, va_gen = imgdg.generate_train_test_images(image_df, train, label)
    print("Length of Test Data Generated : ",len(tt_gen))
    Deep = dp.DeepANN()
    model = Deep.simple_model()
    History = model.fit(
        tr_gen,
        epochs = 10,
        validation_data = va_gen,
        batch_size = 42
    )
    Ann_test_loss, Ann_test_acc = model.evaluate(tt_gen)
    print(f'Test accuracy: {Ann_test_acc}')
    model.save("model.keras")
    print("Summary Of the Model : ")
    print(model.summary())

    image_shape = (28, 28, 3)
    m_adam = Deep.simple_model(image_shape, optimizer='adam')
    m_sgd = Deep.simple_model(image_shape, optimizer='sgd')
    m_rms = Deep.simple_model(image_shape, optimizer='rmsprop')

    models = [m_adam, m_sgd, m_rms]
    Deep.compare_models(models, tr_gen, va_gen, epochs=10)