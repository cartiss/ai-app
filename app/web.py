from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'iejowda32msflsn3dkf7jnad9mk1lpd'


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/tweet-model", methods=["GET", "POST",])
def tweet_model():
    if request.method == 'POST':
        print(request.form)

    return render_template('tweet_model.html', title="Tweet Model")


if __name__ == "__main__":
    app.run(debug=True)