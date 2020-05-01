"""
SVM(Support Vector Machines) are looks like regression solutions.
SVM is the another method to generate a fast model.
SVC(support vector classifier)
SVR(support vector regression)
"""
# higher numbers of correct but not enough

# libraries
import mnist_loader

from sklearn import svm


def svm_main():
    # data
    tr_d, v_d, te_d = mnist_loader.load_data()
    # train
    print("SVM classifier training. Thanks for your patience...")
    classifier = svm.SVC(gamma='scale')
    classifier.fit(tr_d[0], tr_d[1]) # image <-> label
    # test
    predictions = [int(c) for c in classifier.predict(te_d[0])]
    test = [int(classified_image == label) for classified_image, label in zip(predictions, te_d[1])]
    num_correct = sum(test)
    print("SVM classifier results:")
    print(f"{num_correct} of {len(te_d[1])} values correct.")


if __name__ == "__main__":
    svm_main()