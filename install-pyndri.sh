wget https://sourceforge.net/projects/lemur/files/lemur/indri-5.11/indri-5.11.tar.gz/download &&
mv download indri-5.11.tar.gz &&
tar xzvf indri-5.11.tar.gz &&
cd indri-5.11 &&
./configure CXX="g++ -D_GLIBCXX_USE_CXX11_ABI=0" &&
make &&
sudo make install &&
sudo apt install g++ zlib1g-dev python3.5-dev &&
pip install pyndri &&
rm -r indri-5.11 &&
rm indri-5.11.tar.gz
