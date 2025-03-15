from flask import Flask, request
from flask_restful import Api,Resource, reqparse, abort
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
api = Api(app)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///database.db'
db = SQLAlchemy(app)

class VideoModel(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    name = db.Column(db.String(100),nullable=False)
    views = db.Column(db.Integer,nullable=False)
    likes = db.Column(db.Integer,nullable=False)

    def __repr__(self):
        return f"Video(name={name},views={views},likes={likes})"


db.create_all()


video_put_args = reqparse.RequestParser()
video_put_args.add_argument("name",type=str,help="Name of video is required",required=True)
video_put_args.add_argument("views",type=int,help="views of the video",required=True)
video_put_args.add_argument("likes",type=int,help="likes on the video",required=True)

videos = {}
def abort_if_video_id_doesnt_exist(video_id):
    if video_id not in videos:
        abort(404,message="Could not find video...")

def abort_if_video_exists(video_id):
    if video_id in videos:
        abort(409,message="Video already exists with that ID...")