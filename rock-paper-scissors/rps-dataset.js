class RPSDataset {
    constructor() {
        this.labels = [];
    }

    addExample(example, label) {
        if (this.xs == null) {
            //tf.keep saves a tensor from being destroyed by tf.tidy()
            this.xs = tf.keep(example);
        } else {
            const oldX = this.xs;
            this.xs = tf.keep(oldX.concat(example, 0));
            oldX.dispose();
        }
        this.labels.push(label);
    }

    encodeLabels(numClasses) {
        for (var i = 0; i < this.labels.length; i++) {
            if (this.ys == null) {
                this.ys = tf.keep(tf.tidy(
                    () => {
                        return tf.oneHot(
                            tf.tensor1d([this.labels[i]]).toInt(), numClasses)
                    }));
            } else {
                const y = tf.tidy(
                    () => {
                        return tf.oneHot(
                            tf.tensor1d([this.labels[i]]).toInt(), numClasses)
                    });
                const oldY = this.ys;
                this.ys = tf.keep(oldY.concat(y, 0));
                oldY.dispose();
                y.dispose();
            }
        }

    }
}