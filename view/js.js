function getCoords() {
  var start = document.getElementById("start").value;
  var dest = document.getElementById("dest").value;
  var loc = document.getElementById("loc").value;

  // Make an API call to Flask app
  fetch('/route?start=' + start + '&dest=' + dest + '&loc=' + loc, {
    method: 'GET'
  })
  .then(response => response.json())
  .then(data => {
    // Use data received from Flask app to generate map or route
    console.log(data);
  })
  .catch((error) => {
    console.error('Error:', error);
  });
}
