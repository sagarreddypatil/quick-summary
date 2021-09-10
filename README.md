# Quick Summary

This is a monorepo containing all the code for a browser extension to generate summaries of any article with the click of a button.

Quick Note: This is a continuation of sagarreddypatil/summarizer-extension. The older repository is kept frozen as it was used for a Hackathon submission.

## Build Instructions

### Chrome Extension

In order to try out this extension, first clone this repository.

Making sure you have the latest Node.js version available, run the following commands:

1. `cd chrome-extension`
2. `npm install`
3. `npm run build`. This should generate a new `build` folder in the `chrome-extension` directory.
4. Next, open `chrome://extensions` in your browser and click the "Load Unpacked" button. You may need to toggle developer mode on the top right of the page.

  <img width="533" alt="Untitled" src="https://user-images.githubusercontent.com/30810402/93785055-a6155380-fbfb-11ea-9e9e-a6bb592c3b30.png">

5. When your file browser opens, select the new `build` folder that was created. You should now see our extension in the list.

<img width="572" alt="Screen Shot 2020-09-21 at 11 15 39 AM" src="https://user-images.githubusercontent.com/30810402/93785178-c9400300-fbfb-11ea-96a1-aa55c3d5c065.png">

6. And the extension should be installed! To try it out, just press the extension icon on a news article. A summary should load within a few seconds.

### Backend Microservice

1. Make sure you have Docker installed.
1. Clone the repository
1. `cd nlp-flask-app`
1. `docker-compose -f docker-compose.dev.yml up && docker-compose -f docker-compose.dev.yml up`
1. Change line 13 in `./chrome-extension/src/js/popup/utils.js` to use the local server. The URL inside fetch should be changed to something like `http://localhost:5000/summarize` if you are running the chrome extension and the backend on the same server.
   - Do note that the docker image takes a significant amount of time to build as it has to download a large model.
   - The frontend will need to be rebuilt after changing the server address.
1. To use the backend again without rebuilding, just run `docker-compose -f docker-compose.dev.yml up`
1. If you want to run the container headless, run `docker-compose -f docker-compose.dev.yml up -d`
