$(document).ready(function(){
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/twitter');
    var tweets_text = [];
    var tweets_username = [];
    var tweets_label = [];

    //receive details from server
    socket.on('tweets', function(msg) {
        console.log("Tweet_text" + msg.Text);
        console.log("Tweet_username" + msg.Username);
        console.log("Tweet_sentiment" + msg.Sentiment);
        //maintain a list of twenty tweets
        if (tweets_text.length >= 20){
            tweets_text.shift()
        }
        tweets_text.push(msg.Text);
        tweets_username.push(msg.Username);
        tweets_label.push(msg.Sentiment);
        tweets_string = '';

        var pos = 0;
        var neg = 0;
        var neu = 0;

        for (var i = 0; i < tweets_text.length; i++){

            var color = "#28a745";
            
            if(tweets_label[i] == "neutral"){
                color = "#17a2b8";
                neu++;
            }
            if(tweets_label[i] == "negative"){
                color = "#ffc107";
                neg++;
            }
            if(tweets_label[i] == "positive"){
                pos++;
            }

            tweets_string = tweets_string + '<p>' + '@' + tweets_username[i].toString() +
            '</p>' + '<p>' + tweets_text[i].toString() + '</p>' + '<h3 style="color: ' + color + '">' + "Predicted Sentiment: " +
             tweets_label[i].toString() +  '</h3>' ;
        }
        $('#log').html(tweets_string);

        Highcharts.chart('container', {
              chart: {
                  type: 'pie',
                  options3d: {
                      enabled: true,
                      alpha: 45
                  }
              },
              title: {
                  text: 'Twitter Live Feed Sentiment Ratio'
              },
              subtitle: {
                  text: '3D donut'
              },
              plotOptions: {
                  pie: {
                      innerSize: 100,
                      depth: 45
                  }
              },
              series: [{
                  name: 'Sentiment Count',
                  data: [
                      ['Positive', pos],
                      ['Negative', neg],
                      ['Neutral', neu],
                  ]
              }]
          });

        $('#container')

    });

});
