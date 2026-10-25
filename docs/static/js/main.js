// Example-perturbation grid and BibTeX copy button. The page works without JS
// except for the grid, which is built here to keep index.html short.
(function () {
  var datasets = [["caltech", "Caltech-101"], ["aircraft", "FGVC-Aircraft"], ["flowers", "Flowers-102"],
                  ["pet", "Oxford-IIIT Pet"], ["cars", "Stanford Cars"], ["merced", "UC Merced"]];
  var settings = [["clean", "Clean"], ["common3", "Common (s = 3)"],
                  ["l1e75", "ℓ₁ ε = 75"], ["l1e300", "ℓ₁ ε = 300"],
                  ["l2e2", "ℓ₂ ε = 2"], ["l2e8", "ℓ₂ ε = 8"],
                  ["linf4", "ℓ∞ ε = 4/255"], ["linf8", "ℓ∞ ε = 8/255"], ["linf30", "ℓ∞ ε = 30/255"]];
  var body = document.getElementById("grid-body");
  if (body) {
    settings.forEach(function (s) {
      var tr = document.createElement("tr");
      var th = document.createElement("th");
      th.scope = "row"; th.textContent = s[1]; tr.appendChild(th);
      datasets.forEach(function (d) {
        var td = document.createElement("td"), img = document.createElement("img");
        img.src = "static/img/perturbations/" + d[0] + "_" + s[0] + ".jpg";
        img.alt = d[1] + ", " + s[1]; img.loading = "lazy"; img.width = 224; img.height = 224;
        td.appendChild(img); tr.appendChild(td);
      });
      body.appendChild(tr);
    });
  }

  document.querySelectorAll(".copy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = document.getElementById(btn.dataset.target).innerText;
      var done = function () { btn.textContent = "Copied"; setTimeout(function () { btn.textContent = "Copy"; }, 1500); };
      if (navigator.clipboard) { navigator.clipboard.writeText(text).then(done); }
      else {
        var ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta);
        ta.select(); document.execCommand("copy"); document.body.removeChild(ta); done();
      }
    });
  });
})();
