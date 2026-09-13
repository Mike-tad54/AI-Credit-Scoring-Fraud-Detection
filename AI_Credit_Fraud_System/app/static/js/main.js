// Auto-dismiss alerts after 5 seconds.
window.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.alert').forEach(function (a) {
    setTimeout(function () {
      a.classList.remove('show');
      setTimeout(function () { a.remove(); }, 200);
    }, 5000);
  });

  // Show loading spinner on form submit.
  document.querySelectorAll('form').forEach(function (f) {
    f.addEventListener('submit', function () {
      var btn = f.querySelector('button[type="submit"]');
      if (btn && !btn.dataset.original) {
        btn.dataset.original = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<i class="fa fa-spinner fa-spin me-2"></i>Processing...';
      }
    });
  });

  // FR-07: poll fraud-alert notifications for officers/admins.
  function pollNotifications() {
    fetch('/fraud/notifications', { credentials: 'include' })
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (notes) {
        var bell = document.getElementById('notifBell');
        var count = document.getElementById('notifCount');
        var list = document.getElementById('notifList');
        if (!bell) return;
        if (!notes || !notes.length) { bell.style.display = 'none'; return; }
        var unread = notes.filter(function (n) { return !n.is_read; });
        bell.style.display = 'block';
        count.textContent = unread.length;
        count.style.display = unread.length ? 'inline' : 'none';
        list.innerHTML = '<li class="dropdown-header">Fraud Alerts (FR-07) — '
          + notes.length + ' total</li>';
        notes.slice(0, 8).forEach(function (n) {
          var li = document.createElement('li');
          li.innerHTML = '<a class="dropdown-item small ' + (n.is_read ? '' : 'fw-bold') + '">'
            + '<i class="fa fa-triangle-exclamation text-danger me-2"></i>'
            + n.message + '<br><span class="text-muted" style="font-size:.65rem;">'
            + new Date(n.created_at).toLocaleString() + '</span></a>';
          li.addEventListener('click', function () {
            fetch('/fraud/notifications/' + n.notification_id + '/read', {
              method: 'POST', credentials: 'include'
            }).then(function () { pollNotifications(); });
          });
          list.appendChild(li);
        });
        if (notes.length > 8) {
          var more = document.createElement('li');
          more.innerHTML = '<li class="dropdown-item small text-muted text-center">'
            + '+ ' + (notes.length - 8) + ' more</li>';
          list.appendChild(more);
        }
      })
      .catch(function () {});
  }
  pollNotifications();
  setInterval(pollNotifications, 30000);
});

// Toast utility.
function showToast(message, category) {
  category = category || 'info';
  var container = document.querySelector('.container-fluid');
  if (!container) return;
  var div = document.createElement('div');
  div.className = 'alert alert-' + category + ' alert-dismissible fade show';
  div.innerHTML = message + '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>';
  container.insertBefore(div, container.firstChild);
  setTimeout(function () {
    div.classList.remove('show');
    setTimeout(function () { div.remove(); }, 200);
  }, 5000);
}
