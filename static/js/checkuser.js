$(document).ready(function(){
			$.ajax({
                url: "checkuser",
                dataType: "json",
                type:"POST",
                success: function (result) {
                   for(var i=0;i<fform.length;++i)
					{
						var s=fform[i].name;
						var cnt=fform[i].val();
						$("input[name=s]").attr("value",cnt);
					}
                }
            })
			fform=$("#fform");
			$("#submit").submit(function(){
				if(confirm("Are you sure you want to continue?"))
				{
					$.ajax({
				        url: "checkuser",
				        data:JSON.stringify(fform),
				        dataType: "json",
				        type:"POST",
				        success: function (result) {
				        	window.location.href="check.html";
				        }
			    	});
				}
			});
			$("#reset").submit(function(){
			var reason=prompt("Please enter the reason why the audit failed","Please enter in English");
			fform.attr("reason",reason);
			$.ajax({
	        url: "checkuser",
	        data:JSON.stringify(fform),
	        dataType: "json",
	        type:"POST",
	        success: function (result) {
	        	window.location.href="check.html";
	        }
	    });
	});
});