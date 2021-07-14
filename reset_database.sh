sudo docker stop mongodb_sentiment_ja ; sudo docker rm mongodb_sentiment_ja
sudo docker stop redis_sentiment_ja ; sudo docker rm redis_sentiment_ja
cd mongodb ; sudo rm -rf mongodb_data ; mkdir mongodb_data ; sudo chown 1001 ./mongodb_data ; sudo docker-compose up -d ; cd ..
cd redis ; rm -rf redis_data ; mkdir redis_data ; sudo chown 1001 ./redis_data ; sudo docker-compose up -d ; cd ..
sudo docker update --restart unless-stopped mongodb_sentiment_ja
sudo docker update --restart unless-stopped redis_sentiment_ja