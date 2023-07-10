"""Run Flask application."""
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'iejowda32msflsn3dkf7jnad9mk1lpd'


@app.route("/")
def index() -> None:
    """Render home page."""
    return render_template('index.html')


@app.route("/tweet-model", methods=["GET", "POST", ])
def tweet_model() -> None:
    """Render tweet model page."""
    if request.method == 'POST':
        print(request.form)

    return render_template('tweet_model.html', title="Tweet Model")


def main() -> None:
    """Run server."""
    app.run(debug=True)


if __name__ == "__main__":
    main()
