Local build:
https://cloud.google.com/run/docs/building/containers

To build the image use the following command:
<pre>
docker build . --tag gcr.io/aveil-317622/ask_question
</pre>

You then need to allow pushing docker containers to gcloud with the following command:
<pre>
gcloud auth configure-docker
</pre>

To upload the image to cloud run, do the following:
<pre>
docker push gcr.io/aveil-317622/ask_question
</pre>

These steps are done local because the normal method seems to get stuck and fail with docker images that weigh a lot of space.
