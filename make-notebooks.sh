mkdir -p notebooks

cd deep-learning-intro-for-hep

for x in *.md; do y=`echo $x | sed s/.md/.ipynb/`; jupytext --to notebook $x -o ../notebooks/$y --update-metadata '{"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}}'; done

cp -a img ../notebooks/img
cp -a data ../notebooks/data

cd ..
