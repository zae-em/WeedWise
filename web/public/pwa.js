const installSection = document.getElementById("install");
const installButton = document.getElementById("pwa-install-button");
const installHelp = document.getElementById("pwa-install-help");
const installStatus = document.getElementById("pwa-install-status");
const offlineNotice = document.getElementById("pwa-offline");
const standalone = window.matchMedia("(display-mode: standalone)");
let installPrompt;

function updateInstallVisibility() {
  installSection.hidden = standalone.matches || navigator.standalone === true;
}

function updateConnectionNotice() {
  offlineNotice.hidden = navigator.onLine;
}

window.addEventListener("beforeinstallprompt", event => {
  event.preventDefault();
  installPrompt = event;
  updateInstallVisibility();
});

installButton.addEventListener("click", async () => {
  if (!installPrompt) {
    installHelp.open = true;
    installHelp.scrollIntoView({block: "nearest", behavior: "smooth"});
    return;
  }
  const prompt = installPrompt;
  installPrompt = undefined;
  installButton.disabled = true;
  try {
    await prompt.prompt();
    const {outcome} = await prompt.userChoice;
    installStatus.textContent = outcome === "accepted"
      ? "Installation accepted. You can open WeedWise from your apps or home screen."
      : "You can install WeedWise later using your browser menu.";
  } catch {
    installHelp.open = true;
    installStatus.textContent = "Use the browser steps below to install WeedWise.";
  } finally {
    installButton.disabled = false;
  }
});

window.addEventListener("appinstalled", () => {
  installPrompt = undefined;
  installSection.hidden = true;
});
standalone.addEventListener("change", updateInstallVisibility);
window.addEventListener("online", updateConnectionNotice);
window.addEventListener("offline", updateConnectionNotice);
updateInstallVisibility();
updateConnectionNotice();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js", {scope: "./", updateViaCache: "none"})
      .catch(() => {
        // A browser may block service workers; the existing web app remains usable.
      });
  }, {once: true});
}
