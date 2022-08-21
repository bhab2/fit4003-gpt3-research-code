const express = require("express");
const bodyParser = require("body-parser");
const router = express.Router();
const app = express();

const { Configuration, OpenAIApi } = require("openai");
const { response } = require("express");

app.use(bodyParser.urlencoded( { extended: false}));
app.use(bodyParser.json());

router.post('/extract', async (req, res) => {
    //console.log(req.body);
    const { text, apikey } = req.body;
    try {
        const results = await extractKeywords(text, apikey);
        res.send(results.data).end();
    }
    catch(error){
        res.status(400).send(`Could not complete operation. Error dump:\n${error}`).end();
    }

    
})

async function extractKeywords(text, apikey) {
    const configuration = new Configuration({
        apiKey: apikey
      });
      const openai = new OpenAIApi(configuration);
      
      return await openai.createCompletion({
        model: "text-davinci-002",
        prompt: `Extract keywords from this text:n${text}`,
        temperature: 0.3,
        max_tokens: 60,
        top_p: 1.0,
        frequency_penalty: 0.8,
        presence_penalty: 0.0,
      });
}

app.use(router);

app.listen(3000,() => {
console.log("Started on PORT 3000");
})
